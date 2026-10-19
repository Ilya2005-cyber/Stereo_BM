"""Capture a stereo pair on Raspberry Pi and stream it via a local TCP port.

Recommended: carry port 8765 through SSH local port forwarding from Windows.
No root permissions or internet access are required.

Example on Pi:
    python3 stereo_sender_pi.py --left 0 --right 2
"""

import argparse
import socket
import struct
import cv2

PACKET = struct.Struct('!4sIII')  # magic, sequence, JPEG left length, JPEG right length
MAGIC = b'STP1'


def open_camera(index, w, h, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open /dev/video{index}')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    print(f'Camera {index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
          f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, '
          f'FPS request={fps:.1f}', flush=True)
    return cap


def capture_pair(left, right, size):
    ok_l = left.grab()
    ok_r = right.grab()
    if not (ok_l and ok_r):
        raise RuntimeError('Failed to grab from both cameras')
    ok_l, frame_l = left.retrieve()
    ok_r, frame_r = right.retrieve()
    if not (ok_l and ok_r):
        raise RuntimeError('Failed to retrieve from both cameras')
    h, w = frame_l.shape[:2]
    if (w, h) != size or frame_r.shape[:2] != (size[1], size[0]):
        raise RuntimeError(f'Frame size mismatch: left={frame_l.shape}, right={frame_r.shape}, '
                           f'expected={size[0]}x{size[1]}')
    return frame_l, frame_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=int, default=0)
    parser.add_argument('--right', type=int, default=2)
    parser.add_argument('--bind', default='127.0.0.1', help='Default loopback: use SSH tunnel')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=25)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    args = parser.parse_args()
    if args.left == args.right or not (1 <= args.jpeg_quality <= 100):
        parser.error('Check camera indices and JPEG quality (1..100)')

    left = right = None
    try:
        left = open_camera(args.left, args.width, args.height, args.fps)
        right = open_camera(args.right, args.width, args.height, args.fps)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.bind, args.port))
            server.listen(1)
            print(f'Listening on {args.bind}:{args.port}; awaiting Windows client...', flush=True)
            while True:
                conn, addr = server.accept()
                print(f'Connected: {addr}', flush=True)
                with conn:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    seq = 0
                    try:
                        while True:
                            request = conn.recv(1)
                            if not request:
                                break
                            if request != b'N':
                                raise RuntimeError('Unrecognized receiver request')
                            l, r = capture_pair(left, right, (args.width, args.height))
                            opts = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                            ok_l, jpg_l = cv2.imencode('.jpg', l, opts)
                            ok_r, jpg_r = cv2.imencode('.jpg', r, opts)
                            if not (ok_l and ok_r):
                                raise RuntimeError('JPEG compression failed')
                            payload_l, payload_r = jpg_l.tobytes(), jpg_r.tobytes()
                            conn.sendall(PACKET.pack(MAGIC, seq, len(payload_l), len(payload_r)))
                            conn.sendall(payload_l)
                            conn.sendall(payload_r)
                            seq += 1
                            if seq % 50 == 0:
                                print(f'Transmitted {seq} stereo pairs', flush=True)
                    except (BrokenPipeError, ConnectionResetError, EOFError):
                        pass
                print('Disconnected. Waiting for reconnect...', flush=True)
    except KeyboardInterrupt:
        print('\nStopped by Ctrl+C')
    finally:
        if left is not None:
            left.release()
        if right is not None:
            right.release()


if __name__ == '__main__':
    main()
