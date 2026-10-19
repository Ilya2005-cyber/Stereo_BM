r"""Receive paired USB webcam frames from Raspberry Pi via SSH port forwarding.

Windows PowerShell in ONNX-CREStereo-Depth-Estimation:
    & '.\.venv\Scripts\python.exe' '.\stereo_receiver_win.py' --mode preview
    & '.\.venv\Scripts\python.exe' '.\stereo_receiver_win.py' --mode crestereo --distance 98

Pi-side stream carries *unrectified* JPEGs; this script rectifies both frames
using the same calibration.npz as StereoBM. When using JPEG transport, record
that compression may affect measurement accuracy. Both webcams must retain
exactly the same physical L/R calibration orientation as during calibration.
"""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import socket
import struct
import sys
import time

import cv2
import numpy as np

PACKET = struct.Struct('!4sIII')
MAGIC = b'STP1'
ROOT = Path(__file__).resolve().parent
DEFAULT_CALIB = ROOT.parent / 'stereo_project_win' / 'stereo_calibration.npz'
DEFAULT_MODEL = ROOT / 'models' / 'crestereo_combined_iter5_240x320.onnx'
WINDOW = 'Stereo via Raspberry Pi | click ROI in LEFT | S:50 P:save R:center Q:quit'
HEADERS = [
    'timestamp', 'platform', 'transport', 'real_distance_cm', 'measured_distance_cm',
    'std_cm', 'mean_disparity_px', 'absolute_error_cm', 'relative_error_percent',
    'positive_roi_percent', 'mean_inference_ms', 'algorithm_fps',
    'mean_request_to_receive_ms', 'mean_full_cycle_ms', 'full_cycle_fps',
    'samples', 'model', 'model_width', 'model_height', 'roi_x', 'roi_y',
    'roi_size', 'rect_focal_px', 'rect_baseline_calib_units', 'cx_offset_px',
    'rectify_alpha', 'calibration_file', 'provider',
]


def recv_exact(sock, length):
    parts = []
    remaining = length
    while remaining:
        chunk = sock.recv(min(remaining, 262144))
        if not chunk:
            raise ConnectionError('Pi closed the connection during image transfer')
        parts.append(chunk)
        remaining -= len(chunk)
    return b''.join(parts)


def next_pair(sock):
    began = time.perf_counter()
    sock.sendall(b'N')  # Stop-and-wait: no growing queue of old frames
    magic, seq, len_l, len_r = PACKET.unpack(recv_exact(sock, PACKET.size))
    if magic != MAGIC or not (0 < len_l < 5_000_000 and 0 < len_r < 5_000_000):
        raise RuntimeError('Invalid stereo packet from Raspberry Pi')
    left = cv2.imdecode(np.frombuffer(recv_exact(sock, len_l), np.uint8), cv2.IMREAD_COLOR)
    right = cv2.imdecode(np.frombuffer(recv_exact(sock, len_r), np.uint8), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError('JPEG decoding failed')
    return seq, left, right, (time.perf_counter() - began)


def load_rectification(path, alpha):
    with np.load(path) as cal:
        need = ('K_left', 'D_left', 'K_right', 'D_right', 'R', 'T', 'image_size')
        for name in need:
            if name not in cal.files:
                raise RuntimeError(f'Calibration is missing {name}')
        k1, d1 = np.asarray(cal['K_left'], np.float64), np.asarray(cal['D_left'], np.float64)
        k2, d2 = np.asarray(cal['K_right'], np.float64), np.asarray(cal['D_right'], np.float64)
        r, t = np.asarray(cal['R'], np.float64), np.asarray(cal['T'], np.float64)
        size = tuple(int(x) for x in cal['image_size'].ravel()[:2])
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, size, r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    map_l = cv2.initUndistortRectifyMap(k1, d1, r1, p1, size, cv2.CV_16SC2)
    map_r = cv2.initUndistortRectifyMap(k2, d2, r2, p2, size, cv2.CV_16SC2)
    f = float(p1[0, 0])
    b = abs(float(p2[0, 3] / p2[0, 0]))
    offset = float(p1[0, 2] - p2[0, 2])
    if f <= 0 or b <= 0:
        raise RuntimeError('Invalid rectification scale')
    return size, map_l, map_r, f, b, offset


def estimate(disparity, roi, f, b, cx_offset, units_per_cm):
    x, y, radius = roi
    h, w = disparity.shape
    d = disparity[max(0, y-radius):min(h, y+radius+1),
                  max(0, x-radius):min(w, x+radius+1)]
    good = np.isfinite(d) & ((d-cx_offset) > 0.5)
    positive = 100.0 * good.mean() if d.size else 0.0
    if good.sum() < max(50, int(0.2 * d.size)):
        return None, None, positive
    value = float(np.median(d[good]))
    distance = f * b / (value - cx_offset) / units_per_cm
    if not np.isfinite(distance) or distance <= 0:
        return None, value, positive
    return distance, value, positive


def init_crestereo(model):
    # The cuDNN frontend dynamically loads further libraries during inference.
    nvidia_root = Path(sys.executable).resolve().parent.parent / 'Lib' / 'site-packages' / 'nvidia'
    for pkg in ('cudnn', 'cublas', 'cuda_runtime', 'cufft', 'cuda_nvrtc'):
        folder = nvidia_root / pkg / 'bin'
        if folder.is_dir():
            os.environ['PATH'] = str(folder) + os.pathsep + os.environ.get('PATH', '')
    import onnxruntime as ort
    ort.preload_dlls(directory='')
    from crestereo import CREStereo
    net = CREStereo(str(model))
    print('ONNX providers:', net.session.get_providers(), flush=True)
    if 'CUDAExecutionProvider' not in net.session.get_providers():
        raise RuntimeError('CUDAExecutionProvider is missing; do not benchmark on CPU by accident')
    return net


def snapshot(folder, tag, left, right, disparity, view):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f'{tag}_left_rect.png'), left)
    cv2.imwrite(str(folder / f'{tag}_right_rect.png'), right)
    cv2.imwrite(str(folder / f'{tag}_view.png'), view)
    if disparity is not None:
        np.save(folder / f'{tag}_disparity_original_px.npy', disparity)
    print('Saved images to', folder, flush=True)


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists() and path.stat().st_size > 0
    if existing:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            if next(csv.reader(f), []) != HEADERS:
                raise RuntimeError('CSV has a different header; rename it first')
    with path.open('a', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=HEADERS)
        if not existing:
            wr.writeheader()
        wr.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('preview', 'crestereo'), default='preview')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--calibration', type=Path, default=DEFAULT_CALIB)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--distance', type=float, default=None)
    p.add_argument('--units-per-cm', type=float, default=10.0,
                   help='Use 10 when calibration square size is in mm')
    p.add_argument('--roi-radius', type=int, default=15)
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--output', type=Path, default=ROOT / 'remote_stereo_results')
    args = p.parse_args()
    if args.units_per_cm <= 0 or args.samples < 1 or args.roi_radius < 4:
        p.error('Invalid units, samples or ROI radius')
    if args.distance is not None and args.distance <= 0:
        p.error('Distance must be positive')
    if not args.calibration.is_file() or (args.mode == 'crestereo' and not args.model.is_file()):
        p.error('Calibration NPZ or CREStereo model not found')

    size, map_l, map_r, focal, baseline, cx_offset = load_rectification(args.calibration, args.alpha)
    w, h = size
    print(f'Using NPZ={args.calibration}\nf={focal:.2f} px, B={baseline:.3f} '
          f'calibration units, cx_delta={cx_offset:.2f} px, alpha={args.alpha}')
    print('Use the *same physical* L/R cameras and order as Windows calibration.')
    net = init_crestereo(args.model) if args.mode == 'crestereo' else None

    state = {'x': w // 2, 'y': h // 2}
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and x < w and y < h:
            state['x'], state['y'] = x, y
            print(f'ROI: x={x}, y={y}')

    cv2.setMouseCallback(WINDOW, click)
    samples = []
    gathering = False
    last_images = None
    recent = []
    last_fps = 0.0
    outcsv = args.output / 'crestereo_pi_offload_win.csv'

    try:
        with socket.create_connection((args.host, args.port), timeout=10) as sock:
            sock.settimeout(30)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f'Connected to stream at {args.host}:{args.port}')
            while True:
                cycle_begin = time.perf_counter()
                seq, l, r, req_receive_sec = next_pair(sock)
                if l.shape[:2] != (h, w) or r.shape[:2] != (h, w):
                    raise RuntimeError(f'Invalid size: {l.shape} and {r.shape}, expected {w}x{h}')
                left = cv2.remap(l, *map_l, cv2.INTER_LINEAR)
                right = cv2.remap(r, *map_r, cv2.INTER_LINEAR)
                disp_original = None
                inference_sec = 0.0
                dist = disp = pos_pct = None

                if net is not None:
                    disp_net = np.asarray(net(left, right), dtype=np.float32)
                    if 'CUDAExecutionProvider' not in net.session.get_providers():
                        raise RuntimeError('ONNX Runtime fell back to CPU')
                    inference_sec = net.inf_time
                    disp_original = cv2.resize(disp_net, size, cv2.INTER_LINEAR) * (w / net.input_width)
                    dist, disp, pos_pct = estimate(
                        disp_original, (state['x'], state['y'], args.roi_radius),
                        focal, baseline, cx_offset, args.units_per_cm)
                    normalized = np.clip(disp_original / 256.0 * 255.0, 0, 255).astype('uint8')
                    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
                    color[(~np.isfinite(disp_original)) | (disp_original <= 0)] = 0
                else:
                    color = right.copy()

                annotated = left.copy()
                x, y, rad = state['x'], state['y'], args.roi_radius
                cv2.rectangle(annotated, (max(0, x-rad), max(0, y-rad)),
                              (min(w-1, x+rad), min(h-1, y+rad)), (0, 0, 255), 2)
                cv2.putText(annotated, 'PI CAMERA LEFT', (10, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                if net is not None:
                    t = f'DIST: {dist:.1f} cm' if dist is not None else 'DIST: invalid'
                    cv2.putText(annotated, t, (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                .7, (0, 0, 255), 2)
                    cv2.putText(annotated, f'CREStereo: {inference_sec*1000:.0f} ms',
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 2)
                cv2.putText(annotated, f'Network+capture: {req_receive_sec*1000:.0f} ms',
                            (10, 121), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 2)
                cv2.putText(annotated, f'FULL: {last_fps:.1f} FPS',
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 2)
                if gathering:
                    cv2.putText(annotated, f'RECORD: {len(samples)}/{args.samples}',
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
                if net is None:
                    cv2.putText(color, 'PI CAMERA RIGHT (verify order)', (10, 27),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
                combined = np.hstack([annotated, color])
                cv2.imshow(WINDOW, combined)
                key = cv2.waitKey(1) & 0xFF
                cycle_sec = time.perf_counter() - cycle_begin
                recent.append(cycle_sec)
                if len(recent) > 20:
                    recent.pop(0)
                last_fps = 1.0 / float(np.mean(recent))
                last_images = (left, right, disp_original, combined)

                if gathering and net is not None:
                    if dist is not None:
                        samples.append((dist, disp, pos_pct, inference_sec, req_receive_sec, cycle_sec))
                    if len(samples) == args.samples:
                        a = np.asarray(samples, dtype=np.float64)
                        avg = float(a[:,0].mean())
                        std = float(a[:,0].std(ddof=1)) if len(a) > 1 else 0.0
                        abserr = abs(avg - args.distance)
                        inf_sec = float(a[:,3].mean())
                        full_sec = float(a[:,5].mean())
                        row = dict(timestamp=datetime.now().isoformat(timespec='seconds'),
                                   platform='Windows (RTX 3060), Pi camera source',
                                   transport='JPEG/TCP via SSH', real_distance_cm=args.distance,
                                   measured_distance_cm=avg, std_cm=std,
                                   mean_disparity_px=float(a[:,1].mean()), absolute_error_cm=abserr,
                                   relative_error_percent=100.0*abserr/args.distance,
                                   positive_roi_percent=float(a[:,2].mean()),
                                   mean_inference_ms=inf_sec*1000,
                                   algorithm_fps=1.0/inf_sec,
                                   mean_request_to_receive_ms=float(a[:,4].mean())*1000,
                                   mean_full_cycle_ms=full_sec*1000, full_cycle_fps=1.0/full_sec,
                                   samples=len(a), model=args.model.name,
                                   model_width=net.input_width, model_height=net.input_height,
                                   roi_x=x, roi_y=y, roi_size=2*rad+1,
                                   rect_focal_px=focal, rect_baseline_calib_units=baseline,
                                   cx_offset_px=cx_offset, rectify_alpha=args.alpha,
                                   calibration_file=str(args.calibration),
                                   provider=';'.join(net.session.get_providers()))
                        append_row(outcsv, row)
                        print(f'RESULT real={args.distance:.2f} cm, estimated={avg:.2f} +/- {std:.3f} cm, '
                              f'error={row["relative_error_percent"]:.2f}%, '
                              f'NN={row["algorithm_fps"]:.2f} FPS, '
                              f'FULL={row["full_cycle_fps"]:.2f} FPS')
                        print(f'CSV: {outcsv}')
                        snapid = datetime.now().strftime('%Y%m%d_%H%M%S')
                        snapshot(args.output, f'pi_{snapid}', *last_images)
                        samples.clear()
                        gathering = False

                if key == ord('r'):
                    state['x'], state['y'] = w//2, h//2
                if key == ord('s'):
                    if net is None or args.distance is None:
                        print('Press S only in crestereo mode with --distance N')
                    elif gathering:
                        print('Already recording')
                    else:
                        gathering, samples = True, []
                        print(f'Start recording {args.samples} frames at {args.distance:g} cm')
                if key == ord('p') and last_images is not None:
                    snapid = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    snapshot(args.output, f'pi_{snapid}', *last_images)
                if key in (ord('q'), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
