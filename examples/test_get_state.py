#!/usr/bin/env python3
"""
Test get_state specifically from the ZMQ server.
"""

import zmq
import json
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test get_state from ZMQ server")
    parser.add_argument("--server_ip", type=str, default="10.42.0.1",
                        help="Server IP address (default: 10.42.0.1)")
    parser.add_argument("--port", type=int, default=5555,
                        help="Server port (default: 5555)")
    args = parser.parse_args()
    
    server_ip = args.server_ip
    port = args.port
    
    print(f"Connecting to tcp://{server_ip}:{port}")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{server_ip}:{port}")
    
    time.sleep(1)
    print("✓ Connected!\n")
    
    try:
        # Send get_state request
        msg = {"type": "get_state"}
        print(f"Sending: {json.dumps(msg)}")
        socket.send_json(msg)
        
        # Wait for response with timeout
        if socket.poll(5000):  # 5 second timeout
            reply = socket.recv_json()
            print(f"\n✓ Received response:")
            print(json.dumps(reply, indent=2))
        else:
            print("✗ ERROR: No response from server (timeout)")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
