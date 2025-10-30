#!/usr/bin/env python3
"""
send_header.py

Send a 26-byte payload: id(8), date(10), time(8) to a specific IP:port via UDP or TCP.

Usage examples:
  python3 protocol_test.py --proto udp --dst 127.0.0.1 --port 5005 --id DEVICE1 --date 2025-10-27 --time 14:05:09
  python3 send_header.py --proto tcp --dst 192.168.1.50 --port 6000 --id RNT12345 --repeat 100 --interval 0.1
"""
import argparse
import socket
import time
import sys

ID_LEN = 8
DATE_LEN = 10
TIME_LEN = 8
TOTAL_LEN = ID_LEN + DATE_LEN + TIME_LEN

def pad_field(s: str, length: int) -> bytes:
    b = s.encode('ascii', errors='ignore')
    if len(b) > length:
        return b[:length]
    return b + b' ' * (length - len(b))

def build_payload(id_str: str, date_str: str, time_str: str) -> bytes:
    id_b = pad_field(id_str, ID_LEN)
    date_b = pad_field(date_str, DATE_LEN)
    time_b = pad_field(time_str, TIME_LEN)
    payload = id_b + date_b + time_b
    if len(payload) != TOTAL_LEN:
        raise RuntimeError(f"payload length mismatch: {len(payload)} != {TOTAL_LEN}")
    return payload

def send_udp(dst: str, port: int, payload: bytes, timeout=2.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    sock.sendto(payload, (dst, port))
    sock.close()

def send_tcp(dst: str, port: int, payload: bytes, timeout=3.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((dst, port))
    sock.sendall(payload)
    # optional: try to read small response
    try:
        sock.settimeout(0.5)
        _ = sock.recv(1024)
    except Exception:
        pass
    sock.close()

def main():
    p = argparse.ArgumentParser(description="Send 26-byte header to target IP:port")
    p.add_argument("--proto", choices=["udp","tcp"], default="udp")
    p.add_argument("--dst", required=True, help="Destination IP or hostname")
    p.add_argument("--port", type=int, required=True, help="Destination port")
    p.add_argument("--id", required=True, help="ID field (8 bytes max)")
    p.add_argument("--date", required=True, help="Date field (10 bytes, e.g. 2025-10-27)")
    p.add_argument("--time", required=True, help="Time field (8 bytes, e.g. 14:05:09)")
    p.add_argument("--repeat", type=int, default=1, help="Number of times to send (default=1)")
    p.add_argument("--interval", type=float, default=0.0, help="Seconds between repeats")
    p.add_argument("--dry-run", action="store_true", help="Print payload in hex/ASCII instead of sending")
    args = p.parse_args()

    try:
        payload = build_payload(args.id, args.date, args.time)
    except Exception as e:
        print("Payload build error:", e)
        sys.exit(1)

    print(f"Built payload ({len(payload)} bytes):")
    # show hex + ascii preview
    print(payload.hex(' '))
    print(payload)
    if args.dry_run:
        print("Dry run: not sending.")
        return

    for i in range(args.repeat):
        try:
            if args.proto == "udp":
                send_udp(args.dst, args.port, payload)
            else:
                send_tcp(args.dst, args.port, payload)
            print(f"[{i+1}/{args.repeat}] Sent to {args.dst}:{args.port} ({args.proto.upper()})")
        except Exception as e:
            print(f"[ERR] send failed: {e}")
        if i < args.repeat - 1 and args.interval > 0:
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
