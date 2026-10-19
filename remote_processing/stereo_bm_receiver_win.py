r"""StereoBM on Windows, receiving stereo pairs from Raspberry Pi over SSH.

Put this file NEXT TO stereo_receiver_win.py in ONNX-CREStereo-Depth-Estimation.
Keep stereo_sender_pi.py running on Pi and the SSH local tunnel active.

Examples (PowerShell):
    & '.\.venv\Scripts\python.exe' '.\stereo_bm_receiver_win.py' --distance 50
    & '.\.venv\Scripts\python.exe' '.\stereo_bm_receiver_win.py' --distance 100

Keyboard in the image window: S = 50 measurements, P = snapshot,
R = ROI center, Q / Escape = quit. Click inside the LEFT image to move ROI.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import socket
import time

import cv2
import numpy as np

# Reuse the exact transport protocol and rectification code of the CREStereo receiver.
# This avoids accidental differences between the two remote experiments.
from stereo_receiver_win import load_rectification, next_pair, snapshot

ROOT = Path(__file__).resolve().parent
WINDOW = 'StereoBM via Raspberry Pi | click ROI LEFT | S:50 P:save R:center Q:quit'
FIELDS = [
    'timestamp', 'platform', 'transport', 'real_distance_cm', 'measured_distance_cm',
    'std_cm', 'mean_disparity_px', 'absolute_error_cm', 'relative_error_percent',
    'valid_percent', 'mean_bm_ms', 'algorithm_fps', 'mean_request_to_receive_ms',
    'mean_full_cycle_ms', 'full_cycle_fps', 'samples', 'attempts', 'min_disparity',
    'num_disparities', 'block_size', 'roi_x', 'roi_y', 'roi_size', 'rect_focal_px',
    'rect_baseline_calib_units', 'cx_offset_px', 'rectify_alpha',
    'calibration_file',
]


def make_parser():
    parser = argparse.ArgumentParser(description='StereoBM with Raspberry Pi camera stream and Windows processing')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--calibration', type=Path, default=ROOT.parent / 'stereo_project_win' / 'stereo_calibration.npz')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Must match the rectification setting of the Windows BM tests')
    parser.add_argument('--distance', type=float, default=None, help='True distance in cm; required to save measurement series')
    parser.add_argument('--units-per-cm', type=float, default=10.0,
                        help='10 when checkerboard square size was measured in mm')
    parser.add_argument('--min-disparity', type=int, default=0)
    parser.add_argument('--num-disparities', type=int, default=256)
    parser.add_argument('--block-size', type=int, default=15)
    parser.add_argument('--roi-radius', type=int, default=15)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--max-attempts', type=int, default=300)
    parser.add_argument('--output', type=Path, default=ROOT / 'remote_stereo_results')
    return parser


def measure_roi(disparity, cx, cy, radius, focal, baseline, cx_offset, units_per_cm, min_disp):
    h, w = disparity.shape
    area = disparity[max(0, cy-radius):min(h, cy+radius+1),
                     max(0, cx-radius):min(w, cx+radius+1)]
    # OpenCV StereoBM uses minDisparity - 1 as the invalid sentinel.
    valid = (np.isfinite(area) & (area >= min_disp) & ((area-cx_offset) > 0.5))
    valid_pct = float(100.0 * valid.mean()) if area.size else 0.0
    if int(valid.sum()) < max(50, int(area.size * 0.2)):
        return None, None, valid_pct
    median_d = float(np.median(area[valid]))
    cm = focal * baseline / (median_d - cx_offset) / units_per_cm
    if not np.isfinite(cm) or cm <= 0:
        return None, median_d, valid_pct
    return float(cm), median_d, valid_pct


def write_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.is_file() and path.stat().st_size > 0
    if existed:
        with path.open('r', encoding='utf-8-sig', newline='') as fp:
            columns = next(csv.reader(fp), [])
        if columns != FIELDS:
            raise ValueError(f'CSV columns do not match: {path}. Rename the older file before saving.')
    with path.open('a', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        if not existed:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.num_disparities <= 0 or args.num_disparities % 16 != 0:
        parser.error('--num-disparities must be positive and divisible by 16')
    if args.block_size < 5 or args.block_size % 2 == 0:
        parser.error('--block-size must be odd and >= 5')
    if args.roi_radius < 4 or args.samples < 1 or args.max_attempts < args.samples:
        parser.error('Invalid ROI size, samples, or max attempts')
    if args.distance is not None and args.distance <= 0:
        parser.error('--distance must be positive')
    if args.units_per_cm <= 0 or not -1.0 <= args.alpha <= 1.0:
        parser.error('Invalid units-per-cm or rectification alpha')
    if not args.calibration.is_file():
        parser.error(f'NPZ file not found: {args.calibration}')

    size, map_l, map_r, focal, baseline, cx_offset = load_rectification(args.calibration, args.alpha)
    w, h = size
    bm = cv2.StereoBM_create(numDisparities=args.num_disparities, blockSize=args.block_size)
    bm.setMinDisparity(args.min_disparity)
    print(f'Using NPZ: {args.calibration}', flush=True)
    print(f'Rectified f={focal:.3f}px, B={baseline:.3f} calibration units, '
          f'cx difference={cx_offset:.3f}px, alpha={args.alpha}', flush=True)
    print(f'StereoBM: minDisparity={args.min_disparity}, numDisparities={args.num_disparities}, '
          f'blockSize={args.block_size}', flush=True)
    print('Pi handles camera capture. StereoBM runs on the Windows CPU, not on the Pi.', flush=True)
    print('JPEG compression and capture/transfer time are included in the full-cycle test.', flush=True)

    state = {'x': w // 2, 'y': h // 2}
    gathering = False
    samples = []
    attempts = 0
    recent_cycle = []
    last_images = None
    csv_path = args.output / 'bm_pi_offload_win.csv'

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < w and 0 <= y < h:
            state['x'], state['y'] = x, y
            print(f'ROI: x={x}, y={y}', flush=True)

    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        with socket.create_connection((args.host, args.port), timeout=10) as sock:
            sock.settimeout(30)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f'Connected: {args.host}:{args.port}', flush=True)
            print('Click ROI on LEFT image. S = 50 measurements, P = save pair, '
                  'R = center ROI, Q = quit.', flush=True)
            if args.distance is None:
                print('Preview only: specify --distance 50 (or 100, 200) to save CSV.', flush=True)

            while True:
                cycle_begin = time.perf_counter()
                seq, left_raw, right_raw, receive_seconds = next_pair(sock)
                if left_raw.shape[:2] != (h, w) or right_raw.shape[:2] != (h, w):
                    raise ValueError(f'Camera frames are not {w}x{h}: '
                                     f'{left_raw.shape} {right_raw.shape}')
                left = cv2.remap(left_raw, *map_l, cv2.INTER_LINEAR)
                right = cv2.remap(right_raw, *map_r, cv2.INTER_LINEAR)
                left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

                bm_begin = time.perf_counter()
                disparity = bm.compute(left_gray, right_gray).astype(np.float32) / 16.0
                bm_seconds = time.perf_counter() - bm_begin
                depth, median_disp, valid_pct = measure_roi(
                    disparity, state['x'], state['y'], args.roi_radius,
                    focal, baseline, cx_offset, args.units_per_cm, args.min_disparity)

                valid_map = np.isfinite(disparity) & (disparity >= args.min_disparity)
                gray = np.clip((disparity - args.min_disparity) / args.num_disparities * 255,
                               0, 255).astype(np.uint8)
                gray[~valid_map] = 0
                color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                label = left.copy()
                x, y, r = state['x'], state['y'], args.roi_radius
                cv2.rectangle(label, (max(0, x-r), max(0, y-r)),
                              (min(w-1, x+r), min(h-1, y+r)), (0, 0, 255), 2)
                txt = f'{depth:.2f} cm' if depth is not None else 'INVALID'
                cv2.putText(label, f'StereoBM: {txt}', (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)
                cv2.putText(label, f'Valid ROI: {valid_pct:.1f}%  d: '
                            f'{median_disp:.2f}px' if median_disp is not None
                            else f'Valid ROI: {valid_pct:.1f}%  d: INVALID',
                            (12, 62), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 255, 255), 2)
                cv2.putText(label, f'BM: {bm_seconds*1000:.1f}ms  '
                            f'Capture+network: {receive_seconds*1000:.0f}ms',
                            (12, 92), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
                current_fps = 1.0 / float(np.mean(recent_cycle)) if recent_cycle else 0.0
                cv2.putText(label, f'FULL: {current_fps:.1f} FPS', (12, 122),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                if gathering:
                    cv2.putText(label, f'RECORD: {len(samples)}/{args.samples}', (12, 152),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                view = np.hstack((label, color))
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                cycle_seconds = time.perf_counter() - cycle_begin
                recent_cycle.append(cycle_seconds)
                if len(recent_cycle) > 20:
                    recent_cycle.pop(0)
                last_images = (left, right, disparity, view)

                if gathering:
                    attempts += 1
                    if depth is not None:
                        samples.append((depth, median_disp, valid_pct, bm_seconds,
                                        receive_seconds, cycle_seconds))
                    if len(samples) == args.samples:
                        array = np.asarray(samples, dtype=np.float64)
                        mean_cm = float(array[:, 0].mean())
                        std_cm = float(array[:, 0].std(ddof=1)) if len(samples) > 1 else 0.0
                        mean_bm = float(array[:, 3].mean())
                        mean_receive = float(array[:, 4].mean())
                        mean_cycle = float(array[:, 5].mean())
                        abs_err = abs(mean_cm - args.distance)
                        row = {
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                            'platform': 'Windows CPU, Pi camera source',
                            'transport': 'JPEG/TCP via SSH',
                            'real_distance_cm': args.distance,
                            'measured_distance_cm': mean_cm,
                            'std_cm': std_cm,
                            'mean_disparity_px': float(array[:, 1].mean()),
                            'absolute_error_cm': abs_err,
                            'relative_error_percent': abs_err / args.distance * 100,
                            'valid_percent': float(array[:, 2].mean()),
                            'mean_bm_ms': mean_bm * 1000,
                            'algorithm_fps': 1 / mean_bm,
                            'mean_request_to_receive_ms': mean_receive * 1000,
                            'mean_full_cycle_ms': mean_cycle * 1000,
                            'full_cycle_fps': 1 / mean_cycle,
                            'samples': len(samples), 'attempts': attempts,
                            'min_disparity': args.min_disparity,
                            'num_disparities': args.num_disparities,
                            'block_size': args.block_size,
                            'roi_x': state['x'], 'roi_y': state['y'],
                            'roi_size': args.roi_radius * 2 + 1,
                            'rect_focal_px': focal,
                            'rect_baseline_calib_units': baseline,
                            'cx_offset_px': cx_offset,
                            'rectify_alpha': args.alpha,
                            'calibration_file': str(args.calibration),
                        }
                        write_csv(csv_path, row)
                        print(f'RESULT: true={args.distance:.2f}cm '
                              f'BM={mean_cm:.2f} +/- {std_cm:.3f}cm  '
                              f'error={row["relative_error_percent"]:.2f}%  '
                              f'valid={row["valid_percent"]:.1f}%  '
                              f'algorithm={row["algorithm_fps"]:.2f} FPS  '
                              f'FULL={row["full_cycle_fps"]:.2f} FPS', flush=True)
                        print(f'CSV: {csv_path}', flush=True)
                        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        snapshot(args.output, f'bm_pi_{stamp}', *last_images)
                        gathering = False
                        samples.clear()
                        attempts = 0
                    elif attempts >= args.max_attempts:
                        print(f'Aborted: {len(samples)}/{args.samples} valid frames '
                              f'after {attempts} attempts. Move ROI to a textured target.', flush=True)
                        gathering = False
                        samples.clear()
                        attempts = 0

                if key == ord('s'):
                    if args.distance is None:
                        print('To save a series, restart with --distance 50', flush=True)
                    elif gathering:
                        print('Already recording', flush=True)
                    else:
                        gathering, samples, attempts = True, [], 0
                        print(f'Recording {args.samples} valid measurements '
                              f'@ real distance {args.distance:g}cm', flush=True)
                elif key == ord('p'):
                    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    snapshot(args.output, f'bm_pi_{stamp}', *last_images)
                elif key == ord('r'):
                    state['x'], state['y'] = w // 2, h // 2
                elif key in (ord('q'), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
