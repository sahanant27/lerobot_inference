#!/usr/bin/env python3
"""
Simple test script: query `get_state` once, then repeatedly send `set_ee_target`
(cpose) commands at a fixed frequency to the ZMQ server. Useful to verify the
controller path without running the full deploy client.

Behavior:
- Connects REQ -> server
- Sends `get_state`, reads `ee_pos` and `ee_rotvec`
- Repeatedly sends `set_ee_target` with `pos = ee_pos + delta`
- Waits for reply each time and prints status

Usage example:
  python3 examples/test_send_cpose.py --server_ip 10.42.0.1 --port 5555 --freq 10 --duration 5 --delta_z 0.01
"""

import argparse
import time
import json
import zmq


def main():
    parser = argparse.ArgumentParser(description="ZMQ cpose tester")
    parser.add_argument("--server_ip", type=str, default="10.42.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--freq", type=float, default=10.0, help="Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds")
    parser.add_argument("--delta_x", type=float, default=0.0)
    parser.add_argument("--delta_y", type=float, default=0.0)
    parser.add_argument("--delta_z", type=float, default=0.01)
    parser.add_argument("--timeout_ms", type=int, default=3000, help="per-request timeout (ms)")
    args = parser.parse_args()

    addr = f"tcp://{args.server_ip}:{args.port}"
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    # Some zmq libs expect ints
    socket.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))

    print(f"Connecting to {addr}...")
    socket.connect(addr)
    time.sleep(0.2)

    try:
        # initial get_state
        print("Requesting state...")
        socket.send_json({"type": "get_state"})
        if socket.poll(args.timeout_ms):
            state = socket.recv_json()
        else:
            print("ERROR: get_state timeout")
            return

        print("State received:\n", json.dumps(state, indent=2))

        ee_pos = state.get("ee_pos")
        ee_rotvec = state.get("ee_rotvec")
        if ee_pos is None or ee_rotvec is None:
            print("ERROR: server response missing ee_pos/ee_rotvec")
            return

        # plan target as current EE pos + delta
        target_base = [float(x) for x in ee_pos]
        rotvec = [float(x) for x in ee_rotvec]

        delta = [args.delta_x, args.delta_y, args.delta_z]

        freq = float(args.freq)
        period = 1.0 / freq if freq > 0 else 0.1
        steps = max(1, int(args.duration * freq))

        print(f"Sending {steps} set_ee_target commands at {freq} Hz (period {period:.3f}s)")

        for i in range(steps):
            t0 = time.time()
            target = [target_base[j] + delta[j] for j in range(3)]
            req = {"type": "set_ee_target", "pos": target, "rotvec": rotvec}
            socket.send_json(req)

            if socket.poll(args.timeout_ms):
                reply = socket.recv_json()
                print(f"[{i+1}/{steps}] reply: {reply}")
            else:
                print(f"[{i+1}/{steps}] ERROR: set_ee_target timed out")
                # don't break immediately; continue to observe behaviour

            # maintain loop rate
            elapsed = time.time() - t0
            to_sleep = period - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        print("Done sending commands.")

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
