#!/usr/bin/env python3
"""
EE-pose tester for the async_jpose pipeline.

Mirrors what FrankaZMQRobot._send_action_ee does on the client side: query
get_state once for the current EE pose, build a target pose with a small delta,
solve IK locally (placo, via lerobot.model.kinematics.RobotKinematics), and
send joint targets via `set_target` at a fixed rate. Optionally exercises the
gripper grasp/open path against the new pylibfranka-backed handlers.

Server side (Computer A) must be running:
    python run_robot.py --robot_ip 172.16.0.2 --control_mode async_jpose --fps 10

Examples:
  # Move EE up 1 cm at 10 Hz for 5 s, no gripper:
  python3 examples/test_send_cpose.py --server_ip 10.42.0.1 --port 5555 \\
      --freq 10 --duration 5 --delta_z 0.01

  # Same, but close gripper at start and open at end:
  python3 examples/test_send_cpose.py --server_ip 10.42.0.1 --port 5555 \\
      --freq 10 --duration 5 --delta_z 0.01 --gripper close,open

Notes:
  - `delta_*` values are interpreted in BASE frame (same convention as O_T_EE).
  - The target pose held constant across the run (only sent each tick to
    track-and-hold). To sweep, layer an iteration index on top.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq

# Re-use the same kinematics + URDF the robot client uses, so behaviour matches.
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation


URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lerobot"
    / "robots"
    / "franka_fr3"
    / "franka_fr3_kinematics.urdf"
)


def build_kinematics() -> RobotKinematics:
    fk_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
    return RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="fr3_hand_tcp",
        joint_names=fk_joint_names,
    )


def zmq_request(sock: zmq.Socket, req: dict, timeout_ms: int) -> dict:
    sock.send_json(req)
    if not sock.poll(timeout_ms):
        raise TimeoutError(f"timed out waiting for reply to {req.get('type')!r}")
    return sock.recv_json()


def main():
    parser = argparse.ArgumentParser(description="EE-pose tester (client-side IK)")
    parser.add_argument("--server_ip", type=str, default="10.42.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--freq", type=float, default=10.0, help="Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds")
    parser.add_argument("--delta_x", type=float, default=0.0)
    parser.add_argument("--delta_y", type=float, default=0.0)
    parser.add_argument("--delta_z", type=float, default=0.01)
    parser.add_argument("--timeout_ms", type=int, default=3000)
    parser.add_argument(
        "--gripper", type=str, default="",
        help="Comma-separated gripper actions to run before / after the motion. "
             "Allowed: 'close', 'open'. Example: '--gripper close,open' will "
             "grasp before the motion and release after.",
    )
    args = parser.parse_args()

    actions = [a.strip() for a in args.gripper.split(",") if a.strip()]
    for a in actions:
        if a not in ("close", "open"):
            raise SystemExit(f"unknown gripper action {a!r}; choose 'close' or 'open'")

    addr = f"tcp://{args.server_ip}:{args.port}"
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))
    print(f"Connecting to {addr}...")
    sock.connect(addr)
    time.sleep(0.2)

    try:
        # 1. Read current state.
        state = zmq_request(sock, {"type": "get_state"}, args.timeout_ms)
        print("State received:\n", json.dumps(state, indent=2))

        ee_pos = state.get("ee_pos")
        ee_rotvec = state.get("ee_rotvec")
        q0 = state.get("q")
        if ee_pos is None or ee_rotvec is None or q0 is None:
            raise SystemExit("server response missing q / ee_pos / ee_rotvec")

        ee_pos = np.array(ee_pos, dtype=float)
        ee_rotvec = np.array(ee_rotvec, dtype=float)
        q0_rad = np.array(q0, dtype=float)

        # 2. Build the target EE pose: same orientation, position offset by delta.
        delta = np.array([args.delta_x, args.delta_y, args.delta_z], dtype=float)
        target_pos = ee_pos + delta
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec(ee_rotvec).as_matrix()
        t_des[:3, 3] = target_pos
        print(f"Target EE pos: {target_pos}, holding rotvec: {ee_rotvec}")

        # 3. Solve IK once (target is constant for this test).
        kin = build_kinematics()
        q_target_deg = kin.inverse_kinematics(np.rad2deg(q0_rad[:7]), t_des)
        q_target_rad = np.deg2rad(np.asarray(q_target_deg, dtype=float))
        print(f"IK solved -> joint target (rad): {q_target_rad}")

        # 4. Optional: pre-motion gripper close.
        if "close" in actions:
            print("Gripper: grasp close...")
            reply = zmq_request(
                sock,
                {"type": "gripper_grasp", "width": 0.0, "speed": 0.1, "force": 5.0,
                 "epsilon_inner": 0.08, "epsilon_outer": 0.08},
                args.timeout_ms,
            )
            print(f"  reply: {reply}")
            time.sleep(0.5)  # let the gripper settle before motion starts

        # 5. Send joint targets at args.freq Hz for args.duration seconds.
        period = 1.0 / args.freq if args.freq > 0 else 0.1
        steps = max(1, int(args.duration * args.freq))
        target_list = q_target_rad.tolist()
        print(f"Sending {steps} set_target commands at {args.freq} Hz "
              f"(period {period:.3f}s)")

        for i in range(steps):
            t0 = time.time()
            try:
                reply = zmq_request(sock, {"type": "set_target", "q": target_list},
                                    args.timeout_ms)
                print(f"[{i+1}/{steps}] reply: {reply}")
            except TimeoutError as e:
                print(f"[{i+1}/{steps}] {e}")

            to_sleep = period - (time.time() - t0)
            if to_sleep > 0:
                time.sleep(to_sleep)

        # 6. Optional: post-motion gripper open.
        if "open" in actions:
            print("Gripper: move open...")
            reply = zmq_request(
                sock,
                {"type": "gripper_move", "width": 0.08, "speed": 0.1},
                args.timeout_ms,
            )
            print(f"  reply: {reply}")

        print("Done.")

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
