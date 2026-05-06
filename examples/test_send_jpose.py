#!/usr/bin/env python3
"""
jpose tester: query get_state once, then repeatedly send set_target (joint
positions) at a fixed frequency. Mirrors test_send_cpose.py.

Usage:
  python3 examples/test_send_jpose.py --server_ip 10.42.0.1 --port 5555 \
      --freq 10 --duration 5 --delta_q4 0.05
"""

import argparse
import json
import time

import zmq


def main():
    parser = argparse.ArgumentParser(description="ZMQ jpose tester")
    parser.add_argument("--server_ip", type=str, default="10.42.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--freq", type=float, default=10.0, help="Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds")
    # Per-joint delta from the joint pose seen at startup (radians).
    parser.add_argument("--delta_q1", type=float, default=0.0)
    parser.add_argument("--delta_q2", type=float, default=0.0)
    parser.add_argument("--delta_q3", type=float, default=0.0)
    parser.add_argument("--delta_q4", type=float, default=0.05)
    parser.add_argument("--delta_q5", type=float, default=0.0)
    parser.add_argument("--delta_q6", type=float, default=0.0)
    parser.add_argument("--delta_q7", type=float, default=0.0)
    parser.add_argument("--timeout_ms", type=int, default=3000)
    args = parser.parse_args()

    addr = f"tcp://{args.server_ip}:{args.port}"
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))

    print(f"Connecting to {addr}...")
    sock.connect(addr)
    time.sleep(0.2)

    try:
        print("Requesting state...")
        sock.send_json({"type": "get_state"})
        if not sock.poll(args.timeout_ms):
            print("ERROR: get_state timeout")
            return
        state = sock.recv_json()
        print("State received:\n", json.dumps(state, indent=2))

        q0 = state.get("q")
        if q0 is None or len(q0) != 7:
            print("ERROR: server response missing q (or wrong length)")
            return
        q0 = [float(x) for x in q0]

        deltas = [
            args.delta_q1, args.delta_q2, args.delta_q3, args.delta_q4,
            args.delta_q5, args.delta_q6, args.delta_q7,
        ]
        target = [q0[i] + deltas[i] for i in range(7)]

        period = 1.0 / args.freq if args.freq > 0 else 0.1
        steps = max(1, int(args.duration * args.freq))
        print(f"Sending {steps} set_target commands at {args.freq} Hz "
              f"(period {period:.3f}s), target={target}")

        for i in range(steps):
            t0 = time.time()
            sock.send_json({"type": "set_target", "q": target})
            if sock.poll(args.timeout_ms):
                reply = sock.recv_json()
                print(f"[{i+1}/{steps}] reply: {reply}")
            else:
                print(f"[{i+1}/{steps}] ERROR: set_target timed out")

            to_sleep = period - (time.time() - t0)
            if to_sleep > 0:
                time.sleep(to_sleep)

        print("Done sending commands.")

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
