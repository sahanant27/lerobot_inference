#!/usr/bin/env python
"""
Continuous rollout runner for a SINGLE policy / checkpoint.

Loads one policy and runs rollouts back-to-back. SpaceMouse buttons drive the
loop:
  - Button 4  : start the next rollout (skipped with --auto_loop)
  - Button 8  : stop the running rollout early
  - Button 12 : mark the just-finished rollout as SUCCESS
  - Button 13 : mark the just-finished rollout as FAILURE
  - Between rollouts, /franka_reset is published so franka_reset_node
    (error_recovery + goto_home + gripper open) brings the arm to a clean state.

This is the trimmed-down counterpart to deploy_robot_client.py --interactive.
No rollout_model_configs.yaml, no WebSocket bridge — just CLI args.

Example:
    python continuous_rollout.py \\
        --checkpoint_path outputs/train/act_franka_fr3_softtoy/checkpoints/last/pretrained_model \\
        --policy_type act \\
        --task "pick up the soft toy and place it in the drawer" \\
        --model_id act_softtoy \\
        --metadata session_2026-05-06 \\
        --action_ee --obs_ee \\
        --fps 10 \\
        --max_rollout_steps 800
"""

import argparse
import logging
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.franka_fr3.config_franka_fr3 import FrankaFR3Config

# Reuse the inference loops from deploy_robot_client.
sys.path.insert(0, str(Path(__file__).parent))
from deploy_robot_client import run_sync_inference, run_async_inference


# ---------- ROS plumbing for SpaceMouse + reset publish ------------------- #


class RolloutLoopNode(Node):
    BUTTON_START = 4
    BUTTON_STOP = 8
    BUTTON_SUCCESS = 12
    BUTTON_FAILURE = 13

    def __init__(self):
        super().__init__("continuous_rollout_node")
        self.create_subscription(Joy, "/spacenav/joy", self._on_joy, 10)
        self.reset_pub = self.create_publisher(Empty, "/franka_reset", 10)

        self.stop_event = threading.Event()
        self.button_pressed = None
        self.awaiting_start = False
        self.awaiting_result = False

    def _on_joy(self, msg: Joy):
        if len(msg.buttons) < 27:
            return

        if msg.buttons[self.BUTTON_STOP] and not self.awaiting_result:
            self.stop_event.set()

        if self.awaiting_result:
            if msg.buttons[self.BUTTON_SUCCESS]:
                self.button_pressed = True
            elif msg.buttons[self.BUTTON_FAILURE]:
                self.button_pressed = False

        if self.awaiting_start and msg.buttons[self.BUTTON_START]:
            self.button_pressed = True

    def wait_for_start_button(self, logger: logging.Logger):
        logger.info(
            f"Press SpaceMouse button {self.BUTTON_START} to start the next "
            "rollout (Ctrl+C to exit)..."
        )
        self.awaiting_start = True
        self.button_pressed = None
        try:
            while self.button_pressed is None:
                time.sleep(0.1)
        finally:
            self.awaiting_start = False

    def wait_for_result_button(self, logger: logging.Logger) -> str:
        logger.info(
            f"Press SpaceMouse button {self.BUTTON_SUCCESS} (success) or "
            f"button {self.BUTTON_FAILURE} (failure)..."
        )
        self.awaiting_result = True
        self.button_pressed = None
        try:
            while self.button_pressed is None:
                time.sleep(0.1)
        finally:
            self.awaiting_result = False
        return "success" if self.button_pressed else "failure"

    def publish_reset(self):
        self.reset_pub.publish(Empty())


# ---------- main loop ----------------------------------------------------- #


def build_robot_config(args) -> FrankaFR3Config:
    cameras = {
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
    robot_config = FrankaFR3Config(
        id=args.robot_id,
        cameras=cameras,
        dt=1.0 / args.fps,
        action_ee=args.action_ee,
        obs_ee=args.obs_ee,
    )
    if args.max_relative_target is not None:
        robot_config.max_relative_target = args.max_relative_target
    return robot_config


def main():
    parser = argparse.ArgumentParser(
        description="Continuous single-model rollout with SpaceMouse control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--policy_type", type=str, required=True,
                        choices=["act", "diffusion", "vqbet", "smolvla", "groot",
                                 "pi0", "pi05", "xvla", "wall_x"])
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None,
                        help="Short label used in the result log filename. "
                             "Defaults to the checkpoint parent dir name.")
    parser.add_argument("--metadata", type=str, default="session",
                        help="Tag for the result log filename.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--actions_per_chunk", type=int, default=16)
    parser.add_argument("--max_rollout_steps", type=int, default=800)
    parser.add_argument("--policy_device", type=str, default="cuda")
    parser.add_argument("--max_relative_target", type=float, default=0.05)
    parser.add_argument("--robot_id", type=str, default="franka_fr3")
    parser.add_argument("--action_ee", action="store_true")
    parser.add_argument("--obs_ee", action="store_true")
    parser.add_argument("--auto_loop", action="store_true",
                        help="Skip the start-button wait and roll out continuously.")
    parser.add_argument("--reset_wait_s", type=float, default=5.0,
                        help="Seconds to wait after publishing /franka_reset.")
    parser.add_argument("--mode", choices=["sync", "async"], default="sync",
                        help="sync: load policy locally. async: connect to a "
                             "running deploy_policy_server.")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080",
                        help="Policy server address (async mode only).")
    parser.add_argument("--chunk_size_threshold", type=float, default=0.5)
    parser.add_argument("--aggregate_fn_name", type=str,
                        default="weighted_average",
                        choices=["weighted_average", "latest_only", "average",
                                 "conservative"])
    args = parser.parse_args()

    # CLI-only fields the inference functions read off args.
    args.use_sync_inference = (args.mode == "sync")
    args.debug_visualize_queue_size = False
    args.dry_run = False
    args.interactive = False
    args.rename_map = "{}"

    # Logger
    logger = logging.getLogger("continuous_rollout")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(handler)
    logger.propagate = False

    if not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    if args.model_id is None:
        # Use the closest meaningful dir name (skip generic suffixes).
        parts = Path(args.checkpoint_path).parts
        skip = {"pretrained_model", "last", "checkpoints"}
        args.model_id = next(
            (p for p in reversed(parts) if p and p not in skip),
            "model",
        )

    robot_config = build_robot_config(args)

    logger.info("=" * 70)
    logger.info("Continuous Rollout")
    logger.info("=" * 70)
    logger.info(f"  model_id    : {args.model_id}")
    logger.info(f"  policy_type : {args.policy_type}")
    logger.info(f"  checkpoint  : {args.checkpoint_path}")
    logger.info(f"  task        : {args.task}")
    logger.info(f"  metadata    : {args.metadata}")
    logger.info(f"  action_ee   : {args.action_ee}")
    logger.info(f"  obs_ee      : {args.obs_ee}")
    logger.info(f"  fps         : {args.fps}")
    logger.info(f"  max_steps   : {args.max_rollout_steps}")
    logger.info(f"  mode        : {args.mode}")
    if args.mode == "async":
        logger.info(f"  server      : {args.server_address}")
    logger.info(f"  auto_loop   : {args.auto_loop}")
    logger.info("=" * 70)

    # Init ROS, spin SpaceMouse node in a background thread.
    rclpy.init()
    rollout_node = RolloutLoopNode()
    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(rollout_node), daemon=True,
    )
    spin_thread.start()

    # Result log
    result_log = Path(f"logs/results/{args.metadata}_{args.model_id}_result.log")
    result_log.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Result log: {result_log}")

    sync_state = None
    async_client = None
    rollout_idx = 0

    try:
        while True:
            rollout_idx += 1

            if rollout_idx > 1 and not args.auto_loop:
                rollout_node.wait_for_start_button(logger)

            logger.info(f"\n==== Rollout #{rollout_idx} starting ====")

            rollout_node.stop_event.clear()
            rollout_args = Namespace(**vars(args))

            try:
                if args.mode == "sync":
                    if sync_state is not None and sync_state.get("policy") is not None:
                        sync_state["policy"].reset()
                    _, sync_state = run_sync_inference(
                        robot_config,
                        Path(args.checkpoint_path),
                        rollout_args,
                        logger,
                        stop_event=rollout_node.stop_event,
                        sync_state=sync_state,
                    )
                else:  # async
                    if async_client is not None:
                        async_client.reset()
                    _, async_client = run_async_inference(
                        robot_config,
                        Path(args.checkpoint_path),
                        rollout_args,
                        logger,
                        stop_event=rollout_node.stop_event,
                        client=async_client,
                    )
            except Exception as e:
                logger.error(f"Rollout failed: {e}", exc_info=True)
                # Still log it, ask for label, then continue.

            logger.info(f"==== Rollout #{rollout_idx} ended ====")

            # Wait for human label
            result = rollout_node.wait_for_result_button(logger)
            with open(result_log, "a") as f:
                f.write(f"rollout_idx: {rollout_idx}\n")
                f.write(f"model_id: {args.model_id}\n")
                f.write(f"checkpoint: {args.checkpoint_path}\n")
                f.write(f"task: {args.task}\n")
                f.write(f"result: {result}\n\n")
            logger.info(f"Logged: rollout #{rollout_idx} -> {result}")

            # Reset arm
            rollout_node.publish_reset()
            logger.info(
                f"Published /franka_reset. Waiting {args.reset_wait_s}s "
                "for arm to home..."
            )
            time.sleep(args.reset_wait_s)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Cleaning up...")
    finally:
        # Disconnect robot / stop async client.
        if sync_state and sync_state.get("robot") is not None:
            try:
                sync_state["robot"].disconnect()
                logger.info("Robot disconnected.")
            except Exception as e:
                logger.warning(f"Error disconnecting robot: {e}")
        if async_client is not None and async_client.running:
            try:
                async_client.stop()
                logger.info("Async client stopped.")
            except Exception as e:
                logger.warning(f"Error stopping async client: {e}")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
