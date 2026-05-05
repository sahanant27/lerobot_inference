#!/usr/bin/env python

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
Franka FR3 ZMQ Policy Deployment Client (Computer B)

Connects to franka_zmq_server / run_robot.py on Computer A and executes a
trained policy (ACT, Diffusion, VQ-BeT, SmolVLA, GROOT, PI0, PI05) either
synchronously (locally) or asynchronously (via policy server).

Computer A runs:
    python run_robot.py --robot_ip 172.16.0.2 --control_mode jpose --fps 10

Computer B (this script) runs:
    python deploy_robot_client.py \
        --robot_server_address 192.168.1.100 \
        --checkpoint_path outputs/train/act_franka_fr3/checkpoints/last/pretrained_model \
        --policy_type act \
        --task "pick up the soft toy"

Example (EE obs + EE actions, cpose mode on server):
    python deploy_robot_client.py \
        --robot_server_address 192.168.1.100 \
        --obs_ee --action_ee \
        --checkpoint_path outputs/train/diffusion_franka_fr3_ee/checkpoints/last/pretrained_model \
        --policy_type diffusion \
        --task "pick and place task"
    # Server must be started with: python run_robot.py --control_mode cpose

Example (async inference):
    python deploy_robot_client.py \
        --server_address 127.0.0.1:8080 \
        --robot_server_address 192.168.1.100 \
        --checkpoint_path outputs/train/act_franka_fr3/checkpoints/last/pretrained_model \
        --policy_type act \
        --task "pick up the soft toy"
"""

import argparse
import json
import logging
from pathlib import Path

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


def main():
    logger = logging.getLogger("deploy_robot_client")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Franka FR3 ZMQ policy deployment client (Computer B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Robot (Computer A) address
    parser.add_argument(
        "--robot_server_address",
        type=str,
        required=True,
        help="IP address of run_robot.py / franka_zmq_server on Computer A.",
    )
    parser.add_argument("--robot_server_port", type=int, default=5555)
    parser.add_argument("--robot_id", type=str, default="franka_fr3")
    parser.add_argument("--zmq_timeout_ms", type=int, default=5000,
                        help="Per-request ZMQ receive timeout in milliseconds.")
    parser.add_argument("--zmq_max_retries", type=int, default=6,
                        help="How many times to retry each ZMQ request.")
    parser.add_argument("--zmq_retry_backoff_s", type=float, default=0.1,
                        help="Sleep duration between ZMQ retries in seconds.")

    # Observation / action space
    parser.add_argument(
        "--obs_ee",
        action="store_true",
        help="Observations are EE pose (pos + rotvec) instead of joint positions.",
    )
    parser.add_argument(
        "--action_ee",
        action="store_true",
        help="Actions are EE targets instead of joint position targets. "
             "Server must be started with --control_mode cpose.",
    )

    # Policy
    parser.add_argument("--policy_type", type=str, default="",
                        choices=["act", "diffusion", "vqbet", "smolvla", "groot", "pi0", "pi05", "xvla", "wall_x", ""])
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--policy_device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="")

    # Control
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--actions_per_chunk", type=int, default=16)
    parser.add_argument("--chunk_size_threshold", type=float, default=0.5)
    parser.add_argument("--aggregate_fn_name", type=str, default="weighted_average",
                        choices=["weighted_average", "latest_only", "average", "conservative"])
    parser.add_argument("--max_rollout_steps", type=int, default=None)
    parser.add_argument("--max_relative_target", type=float, default=0.05)

    # Inference mode
    parser.add_argument("--use_sync_inference", action="store_true",
                        help="Run policy locally (sync). Recommended for diffusion.")

    # Async server address (only used without --use_sync_inference)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080",
                        help="Policy server address (async mode only).")

    # Debug
    parser.add_argument("--debug_visualize_queue_size", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    # Observation remapping
    parser.add_argument("--rename_map", type=str, default="{}",
                        help='JSON dict to remap observation keys, e.g. \'{"observation.images.front_img": "observation.images.camera1"}\'')

    args = parser.parse_args()

    try:
        rename_map = json.loads(args.rename_map)
        if not isinstance(rename_map, dict):
            raise ValueError("rename_map must be a JSON object")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for rename_map: {e}")

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if not args.dry_run and not args.use_sync_inference is False:
        if checkpoint_path and not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not 0.0 <= args.chunk_size_threshold <= 1.0:
        raise ValueError(f"chunk_size_threshold must be in [0, 1], got {args.chunk_size_threshold}")

    # Camera configuration
    base_camera_configs = {
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

    camera_configs = {}
    for hw_name, cfg in base_camera_configs.items():
        full_key = f"observation.images.{hw_name}"
        if full_key in rename_map:
            new_name = rename_map[full_key].split(".")[-1]
            camera_configs[new_name] = cfg
            logger.info(f"Camera renamed: {hw_name!r} → {new_name!r}")
        else:
            camera_configs[hw_name] = cfg

    # Build robot config (ZMQ only)
    from lerobot.robots.franka_fr3.franka_zmq_robot import FrankaZMQConfig
    robot_config = FrankaZMQConfig(
        id=args.robot_id,
        cameras=camera_configs,
        robot_server_address=args.robot_server_address,
        robot_server_port=args.robot_server_port,
        zmq_timeout_ms=args.zmq_timeout_ms,
        zmq_max_retries=args.zmq_max_retries,
        zmq_retry_backoff_s=args.zmq_retry_backoff_s,
        obs_ee=args.obs_ee,
        action_ee=args.action_ee,
    )
    if args.max_relative_target is not None:
        robot_config.max_relative_target = args.max_relative_target

    logger.info("=" * 70)
    logger.info("Franka FR3 ZMQ Policy Deployment")
    logger.info("=" * 70)
    logger.info(f"Robot:        {args.robot_server_address}:{args.robot_server_port}")
    logger.info(f"ZMQ timeout:  {args.zmq_timeout_ms} ms")
    logger.info(f"ZMQ retries:  {args.zmq_max_retries} (backoff {args.zmq_retry_backoff_s:.2f}s)")
    logger.info(f"Obs space:    {'EE' if args.obs_ee else 'Joint'}")
    logger.info(f"Action space: {'EE' if args.action_ee else 'Joint'}")
    logger.info(f"Policy type:  {args.policy_type.upper() or '(not set)'}")
    logger.info(f"Checkpoint:   {checkpoint_path or '(not set)'}")
    logger.info(f"Task:         {args.task or '(not set)'}")
    logger.info(f"FPS:          {args.fps}")
    logger.info(f"Max steps:    {args.max_rollout_steps or 'unlimited'}")
    for name, cfg in camera_configs.items():
        logger.info(f"Camera {name}: {cfg.serial_number_or_name} @ {cfg.width}x{cfg.height}")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("DRY RUN — config validated. Remove --dry_run to connect.")
        return 0

    if args.use_sync_inference:
        logger.info("Mode: SYNC (policy runs locally)")
        ret, sync_state = run_sync_inference(robot_config, checkpoint_path, args, logger)
        if sync_state and sync_state.get("robot"):
            sync_state["robot"].disconnect()
            logger.info("Robot disconnected.")
        return ret
    else:
        logger.info("Mode: ASYNC (policy runs on server)")
        return run_async_inference(robot_config, checkpoint_path, args, logger)[0]


def run_sync_inference(robot_config, checkpoint_path, args, logger,
                       stop_event=None, sync_state=None):
    """Run policy locally (synchronous).

    Args:
        sync_state: Dict with cached {'policy', 'preprocess', 'postprocess',
                    'robot', 'dataset_features'} — pass between rollouts to
                    avoid reloading.
    Returns:
        (return_code, sync_state)
    """
    import time
    import torch
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame, make_robot_action
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.datasets.utils import hw_to_dataset_features
    from lerobot.utils.constants import ACTION, OBS_STR

    device = torch.device(args.policy_device)

    if sync_state is None:
        sync_state = {}

    policy     = sync_state.get("policy")
    preprocess = sync_state.get("preprocess")
    postprocess = sync_state.get("postprocess")
    robot      = sync_state.get("robot")
    dataset_features = sync_state.get("dataset_features")

    if policy is None:
        logger.info(f"Loading {args.policy_type} policy from {checkpoint_path} ...")
        policy = get_policy_class(args.policy_type).from_pretrained(str(checkpoint_path))
        policy.to(device)
        policy.eval()

        device_override = {"device": device}
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            pretrained_path=str(checkpoint_path),
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )
        _add_resize_processor_if_needed(preprocess, policy.config, robot_config, logger)

    if robot is None:
        logger.info("Connecting to robot ...")
        robot = make_robot_from_config(robot_config)
        robot.connect()
        logger.info("Robot connected.")
        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features    = hw_to_dataset_features(robot.observation_features, OBS_STR)
        dataset_features = {**action_features, **obs_features}

    if hasattr(policy.config, "n_action_steps"):
        orig = policy.config.n_action_steps
        policy.config.n_action_steps = min(args.actions_per_chunk, orig)
        logger.info(f"n_action_steps: {orig} → {policy.config.n_action_steps}")

    logger.info("Control loop started (Ctrl+C to stop).")

    try:
        dt = 1.0 / args.fps
        step = 0
        while args.max_rollout_steps is None or step < args.max_rollout_steps:
            if stop_event and stop_event.is_set():
                logger.info("Stop signal received — ending rollout.")
                break

            t0 = time.perf_counter()

            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, task=args.task,
                ds_features=dataset_features, device=device,
            )
            obs = preprocess(obs_frame)
            action = policy.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            robot.send_action(action)

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

            step += 1

        if args.max_rollout_steps is not None:
            logger.info(f"Completed {step} rollout steps.")

    except KeyboardInterrupt:
        logger.info("Stopping ...")

    sync_state = {
        "policy": policy,
        "preprocess": preprocess,
        "postprocess": postprocess,
        "robot": robot,
        "dataset_features": dataset_features,
    }
    return 0, sync_state


def run_async_inference(robot_config, checkpoint_path, args, logger,
                        stop_event=None, client=None):
    """Run policy on a remote server (asynchronous).

    Note: policies with n_obs_steps > 1 (DP, VQ-BeT) are not fully supported.

    Returns:
        (return_code, client)
    """
    import threading
    import time
    from lerobot.async_inference.helpers import visualize_action_queue_size

    owns_client = client is None
    if client is None:
        from lerobot.async_inference.configs import RobotClientConfig
        from lerobot.async_inference.robot_client import RobotClient

        client_config = RobotClientConfig(
            robot=robot_config,
            server_address=args.server_address,
            policy_type=args.policy_type,
            pretrained_name_or_path=str(checkpoint_path),
            policy_device=args.policy_device,
            task=args.task,
            actions_per_chunk=args.actions_per_chunk,
            chunk_size_threshold=args.chunk_size_threshold,
            fps=args.fps,
            aggregate_fn_name=args.aggregate_fn_name,
            debug_visualize_queue_size=args.debug_visualize_queue_size,
            max_rollout_steps=args.max_rollout_steps,
        )
        client = RobotClient(client_config)

    try:
        if owns_client:
            logger.info("Connecting to policy server and robot ...")
            if not client.start():
                logger.error("Failed to start robot client")
                return 1, client
            logger.info("Connected. Starting control loop ...")

        action_receiver = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver.start()

        if stop_event:
            def _monitor():
                while client.running:
                    if stop_event.is_set():
                        client.shutdown_event.set()
                        break
                    time.sleep(0.1)
            threading.Thread(target=_monitor, daemon=True).start()

        client.control_loop(args.task)

        logger.info("Control loop completed.")
        if owns_client and client.running and stop_event is None:
            client.stop()
        else:
            client.shutdown_event.set()
        action_receiver.join(timeout=5.0)

        if args.debug_visualize_queue_size and hasattr(client, "action_queue_size"):
            visualize_action_queue_size(client.action_queue_size)

    except KeyboardInterrupt:
        logger.info("Stopping ...")
        if client.running and stop_event is None:
            client.stop()
        action_receiver.join(timeout=5.0)
        if args.debug_visualize_queue_size and hasattr(client, "action_queue_size"):
            visualize_action_queue_size(client.action_queue_size)
    except Exception as e:
        logger.error(f"Client error: {e}")
        if client.running and stop_event is None:
            client.stop()
        return 1, client

    return 0, client


def _add_resize_processor_if_needed(preprocessor, policy_config, robot_config, logger):
    """Insert an image resize step if camera dims don't match policy expectations."""
    from lerobot.processor.hil_processor import ImageCropResizeProcessorStep

    if any(isinstance(s, ImageCropResizeProcessorStep) for s in preprocessor.steps):
        return

    expected_dims = {}
    for key, feature in policy_config.input_features.items():
        if feature.type == "VISUAL" and "images" in key:
            cam_name = key.split(".")[-1]
            expected_dims[cam_name] = (feature.shape[1], feature.shape[2])

    needs_resize = False
    for cam_name, cam_cfg in robot_config.cameras.items():
        if cam_name in expected_dims:
            exp_h, exp_w = expected_dims[cam_name]
            if (cam_cfg.height, cam_cfg.width) != (exp_h, exp_w):
                logger.info(
                    f"Camera {cam_name!r}: actual {cam_cfg.width}x{cam_cfg.height} "
                    f"→ expected {exp_w}x{exp_h}"
                )
                needs_resize = True

    if not needs_resize:
        return

    resize_size = next(iter(expected_dims.values()))
    resize_step = ImageCropResizeProcessorStep(resize_size=resize_size)

    insert_idx = next(
        (i + 1 for i, s in enumerate(preprocessor.steps)
         if getattr(s.__class__, "_registry_name", "") == "to_batch_processor"),
        0,
    )
    preprocessor.steps.insert(insert_idx, resize_step)
    logger.info(f"Inserted resize processor at index {insert_idx}, size {resize_size}")


if __name__ == "__main__":
    exit(main())
