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
Franka ZMQ Robot — runs on Computer B (GPU/camera computer)

Drop-in replacement for FrankaFR3 that talks to franka_zmq_server.py on
Computer A (robot computer) via ZMQ instead of ROS2.

Cameras remain local (attached to Computer B). Joint state, EE pose, and
control targets cross the network via ZMQ.

Usage in deploy_robot_client.py:
    python deploy_robot_client.py \
        --robot_server_address 192.168.1.100 \
        --checkpoint_path ... \
        --policy_type act \
        --task "..."
"""

import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras import CameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.rotation import Rotation

from ..config import RobotConfig
from ..robot import Robot

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("franka_zmq")
@dataclass
class FrankaZMQConfig(RobotConfig):
    """Configuration for FrankaZMQRobot (Computer B side of two-computer setup).

    Cameras are read locally. Joint state and targets go to/from franka_zmq_server
    on Computer A via ZMQ.
    """

    # Network address of franka_zmq_server (Computer A)
    robot_server_address: str = "192.168.1.100"
    robot_server_port: int = 5555
    # ZMQ receive timeout in ms — raised as ConnectionError on expiry
    zmq_timeout_ms: int = 5000
    # Number of retries per request before surfacing a ConnectionError.
    zmq_max_retries: int = 6
    # Sleep between retries (seconds), useful when server is briefly busy.
    zmq_retry_backoff_s: float = 0.1

    # Joint names — must match the policy's expected feature keys
    joint_names: list[str] = field(default_factory=lambda: [
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7", "gripper",
    ])

    # EE names: 6-DOF axis-angle pose + gripper (must match policy training)
    ee_names: list[str] = field(default_factory=lambda: [
        "x", "y", "z", "wx", "wy", "wz", "gripper",
    ])
    # Whether observations are EE pose (True) or joint positions (False)
    obs_ee: bool = False
    # Whether actions are EE targets (True) or joint position targets (False)
    action_ee: bool = False

    # IK config (only used when action_ee=True). URDF path defaults to the bundled
    # franka_fr3_kinematics.urdf next to this file.
    urdf_path: str | None = None
    ik_target_frame: str = "fr3_hand_tcp"
    ik_position_weight: float = 1.0
    ik_orientation_weight: float = 0.01

    # Camera configs (same as FrankaFR3Config)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Safety: max joint angle delta per send_action call (radians). None = no limit.
    max_relative_target: float | dict[str, float] | None = None

    # Gripper hysteresis — mirrors FrankaFR3Config
    grasp_threshold: tuple[float, float] = (0.039, 0.02)
    numb_duration: float = 2.0


class FrankaZMQRobot(Robot):
    """Franka FR3 robot interface for Computer B in a two-computer inference setup.

    Reads cameras locally via RealSense. Communicates joint state and targets
    to franka_zmq_server.py on Computer A via ZMQ REQ-REP.
    """

    config_class = FrankaZMQConfig
    name = "franka_zmq"

    def __init__(self, config: FrankaZMQConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = True

        self.cameras: dict | None = None
        self._zmq_context = None
        self._zmq_socket = None

        # Cached from last get_observation to avoid extra round-trip in send_action
        self._last_q: np.ndarray | None = None

        # Gripper hysteresis state. Initialize timer to construction time so the
        # first close/open is gated by numb_duration (matches FrankaInterface on
        # the main branch).
        self._is_grasped: bool = False
        self._gripper_last_change_time: float = time.monotonic()

        # Kinematics for client-side IK when action_ee=True. Server only ever
        # receives joint targets — EE → joint conversion happens here, mirroring
        # FrankaFR3 (main branch).
        self.kinematics = None
        if self.config.action_ee:
            from lerobot.model.kinematics import RobotKinematics

            urdf_path = self.config.urdf_path or str(
                Path(__file__).parent / "franka_fr3_kinematics.urdf"
            )
            fk_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
            self.kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name=self.config.ik_target_frame,
                joint_names=fk_joint_names,
            )
            logger.info(
                f"Initialized IK kinematics with URDF {urdf_path} "
                f"(target_frame={self.config.ik_target_frame})"
            )

    # ------------------------------------------------------------------
    # Feature descriptors (used by lerobot inference pipeline)
    # ------------------------------------------------------------------

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        if self.config.obs_ee:
            motors = {f"{n}.pos": float for n in self.config.ee_names}
        else:
            motors = {f"{j}.pos": float for j in self.config.joint_names}
            # Add duplicate gripper for checkpoint compatibility (9 dims: 7 joints + 2 gripper)
            motors["gripper_duplicate.pos"] = float
        cameras = {
            name: (cfg.height, cfg.width, 3)
            for name, cfg in self.config.cameras.items()
        }
        front_shape = cameras.get("front_img")
        wrist_shape = cameras.get("wrist_img")
        if front_shape is not None:
            cameras["base_0_rgb"] = front_shape
        if wrist_shape is not None:
            cameras["left_wrist_0_rgb"] = wrist_shape
            cameras["right_wrist_0_rgb"] = wrist_shape
        return {**motors, **cameras}

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.action_ee:
            return {f"{n}.pos": float for n in self.config.ee_names}
        return {f"{j}.pos": float for j in self.config.joint_names}

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self._zmq_socket is not None
            and self.cameras is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        self._is_calibrated = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = False) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting FrankaZMQRobot...")

        # 1. Initialize cameras (local — same pattern as FrankaFR3)
        from lerobot.cameras.utils import make_cameras_from_configs
        self.cameras = make_cameras_from_configs(self.config.cameras)
        for cam in self.cameras.values():
            cam.connect()
        logger.info(f"Connected {len(self.cameras)} cameras: {list(self.cameras)}")

        # 2. Open ZMQ REQ socket to franka_zmq_server on Computer A
        self._open_zmq_socket()

        # 3. Verify connection with a test get_state
        try:
            state = self._zmq_get_state()
            logger.info(
                f"Connected to franka_zmq_server at "
                f"{self.config.robot_server_address}:{self.config.robot_server_port} — "
                f"initial q={[f'{v:.3f}' for v in state['q']]}"
            )
        except Exception as e:
            self._close_zmq_socket()
            for cam in self.cameras.values():
                cam.disconnect()
            raise ConnectionError(
                f"Cannot reach franka_zmq_server at "
                f"{self.config.robot_server_address}:{self.config.robot_server_port}: {e}"
            ) from e

        self._is_connected = True
        logger.info("FrankaZMQRobot connected.")

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        logger.info("Disconnecting FrankaZMQRobot...")
        if self.cameras:
            for cam in self.cameras.values():
                cam.disconnect()
        self._close_zmq_socket()
        self._is_connected = False
        logger.info("FrankaZMQRobot disconnected.")

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # One ZMQ round-trip gets both joint state and EE pose
        state = self._zmq_get_state()
        q = state["q"]          # list of 7 floats (arm joints, radians)
        gripper_w = state["gripper"]

        self._last_q = np.array(q)

        obs: dict[str, Any] = {}

        if self.config.obs_ee:
            # EE pose observation (6-DOF axis-angle + gripper)
            for i, name in enumerate(self.config.ee_names[:3]):   # x, y, z
                obs[f"{name}.pos"] = float(state["ee_pos"][i])
            for i, name in enumerate(self.config.ee_names[3:6]):  # wx, wy, wz
                obs[f"{name}.pos"] = float(state["ee_rotvec"][i])
            obs[f"{self.config.ee_names[6]}.pos"] = float(gripper_w)
        else:
            # Joint space observation (7 joints + gripper, plus duplicate gripper for checkpoint compatibility)
            for i, joint in enumerate(self.config.joint_names[:7]):
                obs[f"{joint}.pos"] = float(q[i])
            obs[f"{self.config.joint_names[7]}.pos"] = float(gripper_w)
            # Add duplicate gripper value for checkpoint expecting 9 dims (7 joints + 2 gripper)
            obs["gripper_duplicate.pos"] = float(gripper_w)

        # Camera images from local RealSense
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        # if not self.config.obs_ee:
        if "front_img" in obs:
            obs["base_0_rgb"] = obs["front_img"]
        if "wrist_img" in obs:
            obs["left_wrist_0_rgb"] = obs["wrist_img"]
            obs["right_wrist_0_rgb"] = obs["wrist_img"]

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        if self.config.action_ee:
            return self._send_action_ee(action)
        return self._send_action_joints(action)

    def _send_action_joints(self, action: dict[str, Any]) -> dict[str, Any]:
        # Use cached q from last get_observation to avoid extra round-trip
        if self._last_q is not None:
            current_q = self._last_q.copy()
        else:
            current_q = np.array(self._zmq_get_state()["q"])

        target_q = np.array([
            action.get(f"{j}.pos", current_q[i])
            for i, j in enumerate(self.config.joint_names[:7])
        ])

        # Safety clipping (mirrors FrankaFR3.send_action)
        # NOTE: Should ideally do this in the controller side of things.
        # Also look into the default scale of things.
        if self.config.max_relative_target is not None:
            if isinstance(self.config.max_relative_target, float):
                delta = np.clip(
                    target_q - current_q,
                    -self.config.max_relative_target,
                    self.config.max_relative_target,
                )
                target_q = current_q + delta
            elif isinstance(self.config.max_relative_target, dict):
                for i, joint in enumerate(self.config.joint_names[:7]):
                    if joint in self.config.max_relative_target:
                        max_d = self.config.max_relative_target[joint]
                        delta = np.clip(target_q[i] - current_q[i], -max_d, max_d)
                        target_q[i] = current_q[i] + delta

        self._zmq_set_target(target_q.tolist())

        gripper_key = f"{self.config.joint_names[7]}.pos"
        gripper_pos = float(action.get(gripper_key, 0.08))
        self._maybe_send_gripper(gripper_pos)

        sent: dict[str, Any] = {}
        for i, joint in enumerate(self.config.joint_names[:7]):
            sent[f"{joint}.pos"] = float(target_q[i])
        sent[gripper_key] = gripper_pos
        return sent

    def _send_action_ee(self, action: dict[str, Any]) -> dict[str, Any]:
        # Solve IK locally and ship joint targets — the server only handles joint
        # control. Mirrors FrankaFR3.send_action on the main branch.
        if self.kinematics is None:
            raise RuntimeError(
                "action_ee=True but kinematics is not initialized. "
                "This should never happen — check FrankaZMQRobot.__init__."
            )

        pos = [float(action[f"{n}.pos"]) for n in self.config.ee_names[:3]]
        rotvec = [float(action[f"{n}.pos"]) for n in self.config.ee_names[3:6]]

        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        t_des[:3, 3] = pos

        # Initial guess for IK: cached q from last observation, else fresh fetch.
        if self._last_q is not None:
            q_curr_rad = self._last_q.copy()
        else:
            q_curr_rad = np.array(self._zmq_get_state()["q"])

        q_curr_deg = np.rad2deg(q_curr_rad[:7])
        q_target_deg = self.kinematics.inverse_kinematics(
            q_curr_deg,
            t_des,
            position_weight=self.config.ik_position_weight,
            orientation_weight=self.config.ik_orientation_weight,
        )
        q_target_rad = np.deg2rad(np.asarray(q_target_deg, dtype=float))

        # Apply the same per-joint safety clip used in joint-space mode.
        if self.config.max_relative_target is not None:
            if isinstance(self.config.max_relative_target, float):
                delta = np.clip(
                    q_target_rad - q_curr_rad[:7],
                    -self.config.max_relative_target,
                    self.config.max_relative_target,
                )
                q_target_rad = q_curr_rad[:7] + delta
            elif isinstance(self.config.max_relative_target, dict):
                for i, joint in enumerate(self.config.joint_names[:7]):
                    if joint in self.config.max_relative_target:
                        max_d = self.config.max_relative_target[joint]
                        delta = np.clip(q_target_rad[i] - q_curr_rad[i], -max_d, max_d)
                        q_target_rad[i] = q_curr_rad[i] + delta

        self._zmq_set_target(q_target_rad.tolist())

        gripper_key = f"{self.config.ee_names[6]}.pos"
        gripper_pos = float(action.get(gripper_key, 0.08))
        self._maybe_send_gripper(gripper_pos)

        sent: dict[str, Any] = {f"{n}.pos": v for n, v in zip(self.config.ee_names[:3], pos)}
        sent.update({f"{n}.pos": v for n, v in zip(self.config.ee_names[3:6], rotvec)})
        sent[gripper_key] = gripper_pos
        return sent

    def configure(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

    # ------------------------------------------------------------------
    # Gripper hysteresis
    # ------------------------------------------------------------------

    def _maybe_send_gripper(self, gripper_pos: float) -> None:
        now = time.monotonic()
        elapsed = now - self._gripper_last_change_time
        close_thresh, open_thresh = self.config.grasp_threshold

        if (
            not self._is_grasped
            and elapsed > self.config.numb_duration
            and gripper_pos < close_thresh
        ):
            # Closing uses force-controlled grasp (mirrors FrankaFR3.grasp_close).
            self._zmq_gripper_grasp(width=0.0, speed=0.1, force=5.0,
                                    epsilon_inner=0.08, epsilon_outer=0.08)
            self._is_grasped = True
            self._gripper_last_change_time = now
            logger.info("Gripper: closing (grasp)")

        elif (
            self._is_grasped
            and elapsed > self.config.numb_duration
            and gripper_pos > open_thresh
        ):
            # Opening uses position-only move (mirrors FrankaFR3.grasp_open).
            self._zmq_gripper_move(width=0.08, speed=0.1)
            self._is_grasped = False
            self._gripper_last_change_time = now
            logger.info("Gripper: opening (move)")

    # ------------------------------------------------------------------
    # ZMQ helpers — all network I/O lives here
    # ------------------------------------------------------------------

    def _open_zmq_socket(self) -> None:
        import zmq
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.REQ)
        self._zmq_socket.setsockopt(zmq.LINGER, 0)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, self.config.zmq_timeout_ms)
        addr = f"tcp://{self.config.robot_server_address}:{self.config.robot_server_port}"
        self._zmq_socket.connect(addr)
        logger.debug(f"ZMQ REQ socket connected to {addr}")

    def _close_zmq_socket(self) -> None:
        if self._zmq_socket is not None:
            self._zmq_socket.close()
            self._zmq_socket = None
        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

    def _reconnect_zmq(self) -> None:
        """Reconnect ZMQ socket after a timeout to recover from REQ-REP deadlock."""
        logger.warning("ZMQ timeout — reconnecting socket...")
        self._close_zmq_socket()
        self._open_zmq_socket()

    def _zmq_request(self, msg: dict) -> dict:
        """Send a JSON message and receive the reply, with reconnect on timeout."""
        import zmq
        max_retries = max(1, int(self.config.zmq_max_retries))
        for attempt in range(max_retries):
            try:
                self._zmq_socket.send_json(msg)
                return self._zmq_socket.recv_json()
            except zmq.Again:
                logger.warning(
                    f"ZMQ timeout on {msg.get('type')!r} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    self._reconnect_zmq()
                    if self.config.zmq_retry_backoff_s > 0:
                        time.sleep(self.config.zmq_retry_backoff_s)
        raise ConnectionError(
            f"franka_zmq_server not responding after {max_retries} attempts. "
            f"Is it running at {self.config.robot_server_address}:{self.config.robot_server_port}?"
        )

    def _zmq_get_state(self) -> dict:
        return self._zmq_request({"type": "get_state"})

    def _zmq_set_target(self, q: list[float]) -> dict:
        return self._zmq_request({"type": "set_target", "q": q})

    def _zmq_set_ee_target(self, pos: list[float], rotvec: list[float]) -> dict:
        return self._zmq_request({"type": "set_ee_target", "pos": pos, "rotvec": rotvec})

    def _zmq_gripper_move(self, width: float, speed: float = 0.1) -> dict:
        return self._zmq_request({"type": "gripper_move", "width": width, "speed": speed})

    def _zmq_gripper_grasp(
        self,
        width: float,
        speed: float = 0.1,
        force: float = 5.0,
        epsilon_inner: float = 0.08,
        epsilon_outer: float = 0.08,
    ) -> dict:
        return self._zmq_request({
            "type": "gripper_grasp",
            "width": width,
            "speed": speed,
            "force": force,
            "epsilon_inner": epsilon_inner,
            "epsilon_outer": epsilon_outer,
        })
