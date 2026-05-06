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
from scipy.optimize import least_squares

# Re-use the same kinematics + URDF the robot client uses, so behaviour matches.
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation


def solve_ik_scipy(
    kin: RobotKinematics,
    q_seed_rad: np.ndarray,
    target_pose: np.ndarray,
    max_iters: int = 50,
    pos_tol: float = 1e-4,
    seed_weight: float = 0.05,
) -> np.ndarray:
    """Local IK via scipy.least_squares — won't branch-jump.

    Two key differences from placo's IK:
      1. Damped-least-squares-style regularization: a small residual term
         penalizes deviation from the seed, biasing the solver toward staying
         close to the initial joint config. This is what placo's solver lacks.
      2. method='trf' (Trust Region Reflective) handles m<n problems natively
         (we have 6 pose residuals + 7 regularization residuals vs 7 vars; trf
         is also fine when only the 6 pose residuals are used).

    Args:
        kin: RobotKinematics (used only for FK).
        q_seed_rad: 7-DOF starting joint configuration in radians.
        target_pose: 4x4 homogeneous transform of the desired EE pose.
        seed_weight: regularization strength. 0 → behaves like raw FK-fit
            (can branch-jump). 0.05 (default) → mild bias toward seed, still
            tracks EE within sub-mm. 1.0 → strong bias, may not reach target
            for large Cartesian moves.
    Returns:
        7-DOF joint configuration in radians.
    """
    target_R = target_pose[:3, :3]
    target_p = target_pose[:3, 3]

    def residual(q_rad: np.ndarray) -> np.ndarray:
        T = kin.forward_kinematics(np.rad2deg(q_rad))
        pos_err = T[:3, 3] - target_p
        # Orientation error: axis-angle of T_R · target_R^T → 0 when aligned.
        R_err = T[:3, :3] @ target_R.T
        rotvec_err = Rotation.from_matrix(R_err).as_rotvec()
        # Stay-close-to-seed regularization (small weight so it doesn't
        # dominate the pose tracking, big enough to reject far branches).
        seed_err = seed_weight * (q_rad - q_seed_rad)
        return np.concatenate([pos_err, rotvec_err, seed_err])

    result = least_squares(
        residual, q_seed_rad, method="trf",
        max_nfev=max_iters * len(q_seed_rad),
        xtol=pos_tol, ftol=pos_tol,
    )
    return np.asarray(result.x, dtype=float)


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
    parser.add_argument(
        "--ik_position_weight", type=float, default=1.0,
        help="placo IK position-task weight.",
    )
    parser.add_argument(
        "--ik_orientation_weight", type=float, default=1.0,
        help="placo IK orientation-task weight. Low values let the wrist swing "
             "freely to hit the position; raise to lock orientation. Default 1.0 "
             "(equal weight) is a sane starting point — main's 0.01 is very "
             "position-dominant and can land in surprising joint configurations.",
    )
    parser.add_argument(
        "--max_joint_delta", type=float, default=0.5,
        help="Reject IK solutions whose largest per-joint change from the start "
             "exceeds this many radians (~28°). Catches IK jumping to a far "
             "branch instead of tracking the small Cartesian delta.",
    )
    parser.add_argument(
        "--ik_method", choices=["scipy", "placo"], default="scipy",
        help="IK backend. scipy: Levenberg-Marquardt local solver (won't "
             "branch-jump). placo: lerobot's RobotKinematics.inverse_kinematics "
             "(no seed regularization — can jump branches even for small deltas).",
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
        q_curr_deg = np.rad2deg(q0_rad[:7])

        if args.ik_method == "scipy":
            q_target_rad = solve_ik_scipy(kin, q0_rad[:7], t_des)
            q_target_deg = np.rad2deg(q_target_rad)
        else:
            q_target_deg = kin.inverse_kinematics(
                q_curr_deg, t_des,
                position_weight=args.ik_position_weight,
                orientation_weight=args.ik_orientation_weight,
            )
            q_target_deg = np.asarray(q_target_deg, dtype=float)
            q_target_rad = np.deg2rad(q_target_deg)

        # FK round-trip: where does the IK solution actually place the EE?
        ee_after_ik = kin.forward_kinematics(q_target_deg)
        ik_pos = ee_after_ik[:3, 3]
        pos_err = np.linalg.norm(ik_pos - target_pos)
        z_err = ik_pos[2] - target_pos[2]
        q_delta_deg = q_target_deg - q_curr_deg

        print(f"--- IK diagnostics ({args.ik_method}) ---")
        print(f"  start  EE pos       : {ee_pos}")
        print(f"  target EE pos       : {target_pos}  (delta = {delta})")
        print(f"  FK(IK) EE pos       : {ik_pos}")
        print(f"  position error norm : {pos_err:.4f} m  (Z err: {z_err:+.4f} m)")
        print(f"  per-joint Δ (deg)   : {np.round(q_delta_deg, 2)}")
        print(f"  max |Δ| (rad)       : {np.max(np.abs(np.deg2rad(q_delta_deg))):.3f}")

        max_delta_rad = float(np.max(np.abs(np.deg2rad(q_delta_deg))))
        if max_delta_rad > args.max_joint_delta:
            raise SystemExit(
                f"IK solution moves a joint by {max_delta_rad:.3f} rad "
                f"(> --max_joint_delta {args.max_joint_delta}). "
                "Aborting before sending the target — this is almost certainly "
                "placo jumping to a far IK branch. Try --ik_orientation_weight 5 "
                "or higher, or reduce the EE delta."
            )
        if pos_err > 0.01:
            print(
                f"  WARNING: IK position error {pos_err*1000:.1f} mm exceeds 10 mm — "
                "solver did not converge well. Targets sent to robot anyway."
            )

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