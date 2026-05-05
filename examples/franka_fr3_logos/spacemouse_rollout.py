#!/usr/bin/env python3

# ---------------------------------------------------------------------------
# Copyright (c) 2025 Anant Sah
# Co-developed with Claude Sonnet 4.6 (Anthropic)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

"""
Spacemouse Interactive Rollout Controller (ZMQ, no ROS2)

Python replacement for deploy_robot_client.py --interactive.
Uses pyspacemouse instead of /spacenav/joy for button input.

Workflow:
  1. CLI menu lets you select a model from rollout_model_configs.yaml
  2. Rollout runs; policy + robot state are cached between runs
  3. Spacemouse buttons control the session:
       STOP_BTN   (default: btn 0) — abort rollout early
       SUCCESS_BTN (default: btn 1) — log success after rollout
       FAIL_BTN    (default: btn 2) — log failure after rollout
       (On a 2-button SpaceMouse, btn 0 = stop/fail, btn 1 = success)
  4. Result written to logs/results/<timestamp>_<model_id>.log
  5. Robot holds its last position between rollouts

Requirements:
    pip install pyspacemouse

Usage:
    python spacemouse_rollout.py \
        --robot_server_address 192.168.1.100 \
        --model_configs rollout_model_configs.yaml

    # With EE obs + EE actions:
    python spacemouse_rollout.py \
        --robot_server_address 192.168.1.100 \
        --obs_ee --action_ee \
        --model_configs rollout_model_configs.yaml

rollout_model_configs.yaml format:
    act_softtoy:
      policy_type: act
      mode: sync
      checkpoint_path: outputs/train/act_softtoy/checkpoints/last/pretrained_model
      task: "pick up the soft toy"

    diffusion_blocks:
      policy_type: diffusion
      mode: sync
      checkpoint_path: outputs/train/diffusion_blocks/checkpoints/last/pretrained_model
      task: "stack the blocks"
"""

import argparse
import logging
import sys
import threading
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("spacemouse_rollout")


# ---------------------------------------------------------------------------
# Spacemouse wrapper
# ---------------------------------------------------------------------------

class SpacemouseReader:
    """Thin wrapper around pyspacemouse with button-edge detection."""

    def __init__(self):
        self._dev = None
        self._prev_buttons: list[int] = []

    def open(self) -> bool:
        try:
            import pyspacemouse
            self._dev = pyspacemouse.open()
            if self._dev is None:
                logger.warning("pyspacemouse.open() returned None — no device found.")
                return False
            logger.info("Spacemouse connected.")
            return True
        except Exception as e:
            logger.warning(f"Could not open spacemouse: {e}")
            return False

    def close(self) -> None:
        try:
            import pyspacemouse
            pyspacemouse.close()
        except Exception:
            pass

    def buttons_pressed(self) -> list[int]:
        """Return indices of buttons that just went from 0 → 1 (rising edge)."""
        try:
            import pyspacemouse
            state = pyspacemouse.read()
        except Exception:
            return []

        if state is None:
            self._prev_buttons = []
            return []

        curr = list(state.buttons) if hasattr(state.buttons, "__iter__") else []
        prev = self._prev_buttons or [0] * len(curr)

        rising = [
            i for i, (p, c) in enumerate(zip(prev, curr))
            if p == 0 and c == 1
        ]
        self._prev_buttons = curr
        return rising

    def any_button_held(self, idx: int) -> bool:
        """Return True if button idx is currently held down."""
        try:
            import pyspacemouse
            state = pyspacemouse.read()
        except Exception:
            return False
        if state is None:
            return False
        buttons = list(state.buttons) if hasattr(state.buttons, "__iter__") else []
        return len(buttons) > idx and bool(buttons[idx])


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class SpacemouseRolloutController:
    def __init__(
        self,
        robot_config,
        model_configs: dict,
        base_args: Namespace,
        results_dir: Path,
        stop_btn: int,
        success_btn: int,
        fail_btn: int,
    ):
        self._robot_config = robot_config
        self._model_configs = model_configs
        self._base_args = base_args
        self._results_dir = results_dir
        self._stop_btn = stop_btn
        self._success_btn = success_btn
        self._fail_btn = fail_btn

        self._sm = SpacemouseReader()
        self._sync_state: dict | None = None   # cached policy + robot between rollouts

    def run(self) -> None:
        """Main interactive loop."""
        sm_ok = self._sm.open()
        if not sm_ok:
            logger.warning(
                "Spacemouse not available — falling back to keyboard-only scoring."
            )

        self._results_dir.mkdir(parents=True, exist_ok=True)
        model_ids = list(self._model_configs.keys())

        print("\n" + "=" * 60)
        print("  Spacemouse Rollout Controller")
        print("  Ctrl+C to quit")
        print("=" * 60)

        try:
            while True:
                model_id = self._select_model(model_ids)
                if model_id is None:
                    break

                cfg = self._model_configs[model_id]
                checkpoint_path = Path(cfg["checkpoint_path"])
                task = cfg.get("task", self._base_args.task)
                mode = cfg.get("mode", "sync")

                if not checkpoint_path.exists():
                    logger.error(f"Checkpoint not found: {checkpoint_path}")
                    continue

                print(f"\n  Model   : {model_id}")
                print(f"  Task    : {task}")
                print(f"  Mode    : {mode}")
                print(f"  Btn {self._stop_btn} → stop early  |  Btn {self._success_btn} → success  |  Btn {self._fail_btn} → failure")
                print()

                rollout_args = Namespace(**vars(self._base_args))
                rollout_args.policy_type = cfg.get("policy_type", "")
                rollout_args.checkpoint_path = str(checkpoint_path)
                rollout_args.task = task

                self._run_rollout(model_id, checkpoint_path, rollout_args, mode)

        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            self._sm.close()
            if self._sync_state and self._sync_state.get("robot"):
                self._sync_state["robot"].disconnect()
                logger.info("Robot disconnected.")

    # ------------------------------------------------------------------

    def _select_model(self, model_ids: list[str]) -> str | None:
        """Show a numbered menu and return the selected model ID, or None to quit."""
        print("\nAvailable models:")
        for i, mid in enumerate(model_ids, 1):
            cfg = self._model_configs[mid]
            task = cfg.get("task", "")
            print(f"  {i:2d}. {mid:<30s}  {task}")
        print()

        while True:
            try:
                raw = input("Select model number (or 'q' to quit): ").strip()
            except EOFError:
                return None

            if raw.lower() in ("q", "quit", "exit"):
                return None
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(model_ids):
                    return model_ids[idx]
                print(f"  Enter a number between 1 and {len(model_ids)}.")
            except ValueError:
                print("  Invalid input.")

    def _run_rollout(
        self,
        model_id: str,
        checkpoint_path: Path,
        rollout_args: Namespace,
        mode: str,
    ) -> None:
        from deploy_robot_client import run_sync_inference

        stop_event = threading.Event()

        # Monitor spacemouse STOP_BTN in a daemon thread while rollout runs.
        def _watch_stop():
            while not stop_event.is_set():
                if self._stop_btn in self._sm.buttons_pressed():
                    logger.info(f"Spacemouse btn {self._stop_btn} — aborting rollout.")
                    stop_event.set()
                time.sleep(0.05)

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()

        logger.info(f"Starting rollout: {model_id}")
        t_start = time.monotonic()

        if mode == "sync":
            # Reset policy internal state if reusing cached model.
            if self._sync_state and self._sync_state.get("policy"):
                self._sync_state["policy"].reset()

            _, self._sync_state = run_sync_inference(
                self._robot_config, checkpoint_path, rollout_args, logger,
                stop_event=stop_event,
                sync_state=self._sync_state,
            )
        else:
            logger.warning("Only sync mode is supported in spacemouse_rollout. Skipping.")
            stop_event.set()

        duration = time.monotonic() - t_start
        stop_event.set()   # unblock watcher
        watcher.join(timeout=1.0)

        logger.info(f"Rollout finished in {duration:.1f}s.")
        result = self._score_rollout()
        self._log_result(model_id, rollout_args.task, result, duration)

    def _score_rollout(self) -> str:
        """Wait for spacemouse or keyboard input to score the rollout."""
        print("\n  Score this rollout:")
        print(f"    Spacemouse btn {self._success_btn}  or  [s] + Enter  →  success")
        print(f"    Spacemouse btn {self._fail_btn}  or  [f] + Enter  →  failure")
        print(f"    [r] + Enter                              →  skip (no score)")

        # Poll spacemouse and stdin concurrently.
        result_holder: list[str] = []
        done = threading.Event()

        def _watch_buttons():
            while not done.is_set():
                pressed = self._sm.buttons_pressed()
                if self._success_btn in pressed:
                    result_holder.append("success")
                    done.set()
                elif self._fail_btn in pressed:
                    result_holder.append("failure")
                    done.set()
                time.sleep(0.05)

        btn_thread = threading.Thread(target=_watch_buttons, daemon=True)
        btn_thread.start()

        while not done.is_set():
            # Non-blocking stdin check
            try:
                import select
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.readline().strip().lower()
                    if key in ("s", "success"):
                        result_holder.append("success")
                        done.set()
                    elif key in ("f", "fail", "failure"):
                        result_holder.append("failure")
                        done.set()
                    elif key in ("r", "skip", ""):
                        result_holder.append("skipped")
                        done.set()
            except Exception:
                time.sleep(0.1)

        done.set()
        btn_thread.join(timeout=1.0)

        result = result_holder[0] if result_holder else "skipped"
        symbol = {"success": "✓", "failure": "✗", "skipped": "–"}.get(result, "?")
        print(f"  → {symbol}  {result.upper()}")
        return result

    def _log_result(self, model_id: str, task: str, result: str, duration: float) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self._results_dir / f"{timestamp}_{model_id}.log"
        with open(log_path, "w") as f:
            f.write(f"model_id:  {model_id}\n")
            f.write(f"task:      {task}\n")
            f.write(f"result:    {result}\n")
            f.write(f"duration:  {duration:.1f}s\n")
            f.write(f"timestamp: {timestamp}\n")
        logger.info(f"Result logged → {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spacemouse interactive rollout controller (ZMQ, no ROS2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Robot
    parser.add_argument("--robot_server_address", type=str, required=True,
                        help="IP of run_robot.py on Computer A.")
    parser.add_argument("--robot_server_port", type=int, default=5555)
    parser.add_argument("--robot_id", type=str, default="franka_fr3")

    # Observation / action space
    parser.add_argument("--obs_ee", action="store_true")
    parser.add_argument("--action_ee", action="store_true")

    # Model configs
    parser.add_argument(
        "--model_configs",
        type=str,
        default="rollout_model_configs.yaml",
        help="Path to YAML file mapping model IDs to checkpoint paths / tasks.",
    )

    # Results
    parser.add_argument("--results_dir", type=str, default="logs/results",
                        help="Directory to write per-rollout result logs.")

    # Control
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_rollout_steps", type=int, default=None)
    parser.add_argument("--max_relative_target", type=float, default=0.05)
    parser.add_argument("--actions_per_chunk", type=int, default=16)
    parser.add_argument("--policy_device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="",
                        help="Fallback task if not in model_configs YAML.")

    # Spacemouse button mapping
    parser.add_argument("--stop_btn", type=int, default=0,
                        help="Spacemouse button index to abort rollout early.")
    parser.add_argument("--success_btn", type=int, default=1,
                        help="Spacemouse button index to score as success.")
    parser.add_argument("--fail_btn", type=int, default=2,
                        help="Spacemouse button index to score as failure. "
                             "On 2-button SpaceMouse set to 0 (shares with stop).")

    args = parser.parse_args()

    # Load model configs
    model_configs_path = Path(args.model_configs)
    if not model_configs_path.exists():
        logger.error(f"Model configs not found: {model_configs_path}")
        return 1

    with open(model_configs_path) as f:
        model_configs = yaml.safe_load(f)

    if not model_configs:
        logger.error("Model configs file is empty.")
        return 1

    # Build robot config
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.robots.franka_fr3.franka_zmq_robot import FrankaZMQConfig

    camera_configs = {
        "front_img": RealSenseCameraConfig(
            serial_number_or_name="938422074102",
            width=640, height=480, fps=30,
            rotation=Cv2Rotation.ROTATE_180,
        ),
        "wrist_img": RealSenseCameraConfig(
            serial_number_or_name="919122070360",
            width=640, height=480, fps=30,
        ),
    }

    robot_config = FrankaZMQConfig(
        id=args.robot_id,
        cameras=camera_configs,
        robot_server_address=args.robot_server_address,
        robot_server_port=args.robot_server_port,
        obs_ee=args.obs_ee,
        action_ee=args.action_ee,
        max_relative_target=args.max_relative_target,
    )

    # Shared args passed down to run_sync_inference
    base_args = Namespace(
        policy_type="",
        checkpoint_path="",
        policy_device=args.policy_device,
        task=args.task,
        fps=args.fps,
        actions_per_chunk=args.actions_per_chunk,
        max_rollout_steps=args.max_rollout_steps,
        # Unused by sync path but kept for signature compatibility
        chunk_size_threshold=0.5,
        aggregate_fn_name="weighted_average",
        debug_visualize_queue_size=False,
        server_address="",
        use_sync_inference=True,
    )

    controller = SpacemouseRolloutController(
        robot_config=robot_config,
        model_configs=model_configs,
        base_args=base_args,
        results_dir=Path(args.results_dir),
        stop_btn=args.stop_btn,
        success_btn=args.success_btn,
        fail_btn=args.fail_btn,
    )
    controller.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
