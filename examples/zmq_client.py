#!/usr/bin/env python3
"""
CORRECT ZMQ test client — runs on Computer B (policy computer)
CONNECTS to server (does not bind).
"""

import zmq
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="ZMQ test client")
    parser.add_argument("--server_ip", type=str, default="10.42.0.1",
                        help="Server IP address (default: 10.42.0.1)")
    parser.add_argument("--port", type=int, default=5555,
                        help="Server port (default: 5555)")
    args = parser.parse_args()
    
    server_ip = args.server_ip
    port = args.port
    
    print(f"Creating ZMQ context and REQ socket...")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    
    print(f"Connecting to tcp://{server_ip}:{port}")
    socket.connect(f"tcp://{server_ip}:{port}")  # <-- CONNECT, not BIND
    time.sleep(1)
    print("✓ Connected! Sending test messages...\n")
    
    try:
        for i in range(5):
            msg = f"Test message {i+1}"
            print(f"\nSending: {msg}")
            socket.send_string(msg)
            
            # Set timeout to avoid hanging
            if socket.poll(5000):  # 5 second timeout
                reply = socket.recv_string()
                print(f"Received: {reply}")
            else:
                print("ERROR: No response from server (timeout)")
                break
            
            time.sleep(0.5)
        
        print("\n✓ Connection test successful!")
        
    except zmq.error.Again:
        print("ERROR: Timeout waiting for server response")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()