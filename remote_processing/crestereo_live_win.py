r"""Live stereo distance measurement with CREStereo (Windows, OpenCV + ONNX Runtime GPU).

Copy next to image_depth_estimation.py in ONNX-CREStereo-Depth-Estimation.
Example (from this directory, in PowerShell):
    & ".\.venv\Scripts\python.exe" ".\crestereo_live_win.py" --distance 98
    & ".\.venv\Scripts\python.exe" ".\crestereo_live_win.py" --distance 195 --left 0 --right 2

Click a point ON THE TARGET inside the LEFT half of the window.
S: collect 50 valid measurements and append one CSV row
P: save current rectified pair, disparity (NumPy), and visualization
Q / Esc: quit; R: reset ROI center

IMPORTANT: Calibration must match these exact physical cameras, L/R order,
resolution, and stereoRectify settings. Positive disparity fraction in the ROI
is NOT a neural network confidence measure and is NOT comparable to BM's
valid pixel percentage. Camera grabbing is not hardware synchronization.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

# Windows: cuDNN loads some sublibraries only during the first inference.
# Prepend NVIDIA pip-package DLL directories *before* creating the session.
NVIDIA_ROOT = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
for component in ("cudnn", "cublas", "cuda_runtime", "cufft", "cuda_nvrtc"):
    directory = NVIDIA_ROOT / component / "bin"
    if directory.is_dir():
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
try:
    ort.preload_dlls(directory="")
except AttributeError:
    pass
from crestereo import CREStereo  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_NPZ = ROOT.parent / "stereo_project_win" / "stereo_calibration.npz"
DEFAULT_MODEL = ROOT / "models" / "crestereo_combined_iter5_240x320.onnx"
WINDOW = "CREStereo LIVE | click target in LEFT | S:50 P:save R:center Q:quit"

CSV_FIELDS = [
    "timestamp", "platform", "real_distance_cm", "measured_distance_cm", "std_cm",
    "mean_disparity_px", "absolute_error_cm", "relative_error_percent",
    "positive_roi_percent", "mean_inference_ms", "algorithm_fps", "mean_full_cycle_ms",
    "full_cycle_fps", "samples", "attempts", "model", "model_width", "model_height",
    "roi_x", "roi_y", "roi_size", "left_camera", "right_camera",
    "rectify_alpha", "rect_focal_px", "rect_baseline_calib_units", "cx_offset_px",
    "calibration_file", "providers",
]


def options():
    p = argparse.ArgumentParser(description="CREStereo live distance measurement from two USB webcams")
    p.add_argument("--left", type=int, default=0, help="Index of LEFT physical camera (default: 0)")
    p.add_argument("--right", type=int, default=1, help="Index of RIGHT physical camera (default: 1)")
    p.add_argument("--distance", type=float, default=None, help="Measured reference distance to target, cm")
    p.add_argument("--calibration", type=Path, default=DEFAULT_NPZ, help="Windows stereo_calibration.npz")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="CREStereo ONNX model")
    p.add_argument("--alpha", type=float, default=1.0, help="Must equal StereoBM stereoRectify alpha (default 1.0)")
    p.add_argument("--roi-radius", type=int, default=15, help="ROI half-size; 15 -> 31x31 pixels")
    p.add_argument("--calib-units-per-cm", type=float, default=10.0,
                   help="10 if square size is in mm, 1 if in cm")
    p.add_argument("--backend", choices=("dshow", "msmf", "auto"), default="dshow")
    p.add_argument("--camera-fps", type=float, default=25.0)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--max-attempts", type=int, default=300)
    p.add_argument("--output", type=Path, default=ROOT / "crestereo_results")
    p.add_argument("--allow-cpu", action="store_true", help="Allow execution without CUDA (much slower)")
    return p.parse_args()


def load_calibration(npz_path: Path, alpha: float):
    with np.load(npz_path) as data:
        fields = ("K_left", "D_left", "K_right", "D_right", "R", "T", "image_size")
        missing = [key for key in fields if key not in data.files]
        if missing:
            raise ValueError(f"Missing NPZ keys: {missing}")
        k1 = np.asarray(data["K_left"], dtype=np.float64)
        d1 = np.asarray(data["D_left"], dtype=np.float64)
        k2 = np.asarray(data["K_right"], dtype=np.float64)
        d2 = np.asarray(data["D_right"], dtype=np.float64)
        rotation = np.asarray(data["R"], dtype=np.float64)
        translation = np.asarray(data["T"], dtype=np.float64).reshape(3, 1)
        width, height = (int(x) for x in data["image_size"].ravel()[:2])
    size = width, height
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
    )
    map_l = cv2.initUndistortRectifyMap(k1, d1, r1, p1, size, cv2.CV_16SC2)
    map_r = cv2.initUndistortRectifyMap(k2, d2, r2, p2, size, cv2.CV_16SC2)
    focal = float(p1[0, 0])
    baseline = abs(float(p2[0, 3] / p2[0, 0]))
    cx_offset = float(p1[0, 2] - p2[0, 2])
    if focal <= 0 or baseline <= 0:
        raise ValueError("Invalid calibrated rectified focal length or baseline")
    return size, map_l, map_r, focal, baseline, cx_offset


def open_camera(index: int, backend: str, size: tuple[int, int], fps: float):
    api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}[backend]
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera {index} with backend={backend}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def grab_pair(left_cap, right_cap, size):
    # Closely timed software grabs; NOT true hardware synchronization.
    ok_l = left_cap.grab()
    ok_r = right_cap.grab()
    if not (ok_l and ok_r):
        raise RuntimeError("Camera grab failed; check USB ports and camera indices")
    ok_l, left = left_cap.retrieve()
    ok_r, right = right_cap.retrieve()
    if not (ok_l and ok_r):
        raise RuntimeError("Camera retrieve failed")
    if left.shape[:2] != size[::-1] or right.shape[:2] != size[::-1]:
        raise RuntimeError(f"Frame size mismatch. NPZ={size}, left={left.shape[:2]}, right={right.shape[:2]}")
    return left, right


def get_roi(cx: int, cy: int, radius: int, w: int, h: int):
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(w, cx + radius + 1)
    y1 = min(h, cy + radius + 1)
    return x0, y0, x1, y1


def estimate_distance(disp_orig, region, f, b, cx_offset, units_per_cm):
    x0, y0, x1, y1 = region
    roi = disp_orig[y0:y1, x0:x1]
    usable = np.isfinite(roi) & ((roi - cx_offset) > 0.5)
    positive_percent = 100.0 * np.count_nonzero(usable) / max(1, roi.size)
    # At least 50 pixels and 20% of ROI should have a positive denominator.
    if np.count_nonzero(usable) < max(50, int(roi.size * 0.2)):
        return None, None, positive_percent
    median_disp = float(np.median(roi[usable]))
    distance_cm = f * b / (median_disp - cx_offset) / units_per_cm
    if not np.isfinite(distance_cm) or distance_cm <= 0:
        return None, median_disp, positive_percent
    return distance_cm, median_disp, positive_percent


def save_snapshot(output: Path, basename: str, left, right, dmap, display):
    output.mkdir(parents=True, exist_ok=True)
    # Both rectified images can be reused for BM and neural evaluation.
    cv2.imwrite(str(output / f"{basename}_rect_left.png"), left)
    cv2.imwrite(str(output / f"{basename}_rect_right.png"), right)
    cv2.imwrite(str(output / f"{basename}_view.png"), display)
    np.save(output / f"{basename}_disparity_px.npy", dmap.astype(np.float32))
    print(f"Saved rectified pair, disparity array and visualization: {output / basename}*")


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    already_exists = path.is_file() and path.stat().st_size > 0
    if already_exists:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            columns = next(csv.reader(f), [])
        if columns != CSV_FIELDS:
            raise ValueError(f"Unexpected CSV header in {path}; move/rename old file first")
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not already_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = options()
    if args.left == args.right:
        raise ValueError("Left and right indices must differ")
    if not (-1.0 <= args.alpha <= 1.0):
        raise ValueError("alpha must be from -1 to 1")
    if args.roi_radius < 4 or args.samples <= 0 or args.max_attempts < args.samples:
        raise ValueError("Invalid ROI radius or sample count")
    if args.calib_units_per_cm <= 0:
        raise ValueError("calib-units-per-cm must be > 0")
    for path in (args.calibration, args.model):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    real_cm = args.distance
    if real_cm is None:
        answer = input("Reference distance (cm, e.g. 98; Enter for preview only): ").strip().replace(",", ".")
        real_cm = float(answer) if answer else None
    if real_cm is not None and real_cm <= 0:
        raise ValueError("Reference distance must be > 0")

    size, maps_l, maps_r, focal, baseline, cx_offset = load_calibration(args.calibration, args.alpha)
    w, h = size
    print(f"NPZ: {args.calibration}")
    print(f"Rectified focal={focal:.2f} px | baseline={baseline:.3f} calibration units | "
          f"cx_left-cx_right={cx_offset:.3f} px | alpha={args.alpha}")
    print("WARNING: distance in cm assumes the checkerboard square was specified in mm.")
    print("WARNING: left/right order and calibration must match the physical camera pair.")

    net = CREStereo(str(args.model))
    providers = net.session.get_providers()
    print("ONNX providers:", providers)
    if "CUDAExecutionProvider" not in providers and not args.allow_cpu:
        raise RuntimeError("CUDA is not active. Fix GPU DLLs or run explicitly with --allow-cpu")

    left_cap = right_cap = None
    try:
        left_cap = open_camera(args.left, args.backend, size, args.camera_fps)
        right_cap = open_camera(args.right, args.backend, size, args.camera_fps)
        # Warm up GPU with a real rectified frame pair (also validates camera geometry/size).
        l_raw, r_raw = grab_pair(left_cap, right_cap, size)
        warm_l = cv2.remap(l_raw, maps_l[0], maps_l[1], cv2.INTER_LINEAR)
        warm_r = cv2.remap(r_raw, maps_r[0], maps_r[1], cv2.INTER_LINEAR)
        print("Warming up neural network (3 runs)...")
        for _ in range(3):
            net(warm_l, warm_r)
        if "CUDAExecutionProvider" not in net.session.get_providers() and not args.allow_cpu:
            raise RuntimeError("GPU execution fell back to CPU during warmup")

        state = {"cx": w // 2, "cy": h // 2}
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(1280, 1400), min(550, h + 80))

        def on_mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < w and 0 <= y < h:
                state["cx"], state["cy"] = x, y
                print(f"ROI center: x={x}, y={y}")

        cv2.setMouseCallback(WINDOW, on_mouse)
        print("Click a target inside the LEFT image to move ROI; S = 50 readings; P = snapshot; R = center; Q = quit.")
        if real_cm is None:
            print("PREVIEW ONLY: to save measurement series, restart with --distance 98 (or other real distance).")
        else:
            print(f"Reference distance for current session: {real_cm:.2f} cm")

        gathering = False
        samples = []
        attempts = 0
        recent_frame_times = []
        latest_frames = None
        last_full_fps = 0.0
        csv_path = args.output / "crestereo_results_win.csv"

        while True:
            started = time.perf_counter()
            l_raw, r_raw = grab_pair(left_cap, right_cap, size)
            left = cv2.remap(l_raw, maps_l[0], maps_l[1], cv2.INTER_LINEAR)
            right = cv2.remap(r_raw, maps_r[0], maps_r[1], cv2.INTER_LINEAR)
            disp_network = net(left, right)
            if not args.allow_cpu and "CUDAExecutionProvider" not in net.session.get_providers():
                raise RuntimeError("ONNX Runtime switched to CPU during live inference")
            # The network outputs disparity in its own horizontal pixel units.
            disp_orig = cv2.resize(np.asarray(disp_network, dtype=np.float32),
                                   size, interpolation=cv2.INTER_LINEAR) * (w / float(net.input_width))
            reg = get_roi(state["cx"], state["cy"], args.roi_radius, w, h)
            distance, median_disp, positive_pct = estimate_distance(
                disp_orig, reg, focal, baseline, cx_offset, args.calib_units_per_cm
            )
            color = net.draw_disparity()
            left_annotated = left.copy()
            x0, y0, x1, y1 = reg
            cv2.rectangle(left_annotated, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
            distance_text = f"DIST: {distance:.1f} cm" if distance is not None else "DIST: invalid"
            cv2.putText(left_annotated, distance_text, (15, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
            cv2.putText(left_annotated, f"CREStereo: {1000 * net.inf_time:.0f} ms  "
                        f"FULL: {last_full_fps:.1f} FPS", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(left_annotated, f"Positive ROI: {positive_pct:.1f}%", (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if gathering:
                cv2.putText(left_annotated, f"RECORD: {len(samples)}/{args.samples}", (15, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            display = np.hstack((left_annotated, color))
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            duration = time.perf_counter() - started
            recent_frame_times.append(duration)
            if len(recent_frame_times) > 20:
                recent_frame_times.pop(0)
            last_full_fps = 1.0 / np.mean(recent_frame_times)
            latest_frames = (left, right, disp_orig.copy(), display.copy())

            if gathering:
                attempts += 1
                if distance is not None:
                    samples.append((distance, median_disp, positive_pct, net.inf_time, duration))
                if len(samples) >= args.samples:
                    measurements = np.asarray(samples, dtype=float)
                    avg_cm = float(np.mean(measurements[:, 0]))
                    std_cm = float(np.std(measurements[:, 0], ddof=1)) if len(samples) > 1 else 0.0
                    avg_d = float(np.mean(measurements[:, 1]))
                    avg_positive = float(np.mean(measurements[:, 2]))
                    avg_inf_ms = float(np.mean(measurements[:, 3]) * 1000.0)
                    avg_full_ms = float(np.mean(measurements[:, 4]) * 1000.0)
                    abs_error = abs(avg_cm - real_cm)
                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "platform": "Windows", "real_distance_cm": real_cm,
                        "measured_distance_cm": avg_cm, "std_cm": std_cm,
                        "mean_disparity_px": avg_d, "absolute_error_cm": abs_error,
                        "relative_error_percent": 100 * abs_error / real_cm,
                        "positive_roi_percent": avg_positive,
                        "mean_inference_ms": avg_inf_ms, "algorithm_fps": 1000.0 / avg_inf_ms,
                        "mean_full_cycle_ms": avg_full_ms, "full_cycle_fps": 1000.0 / avg_full_ms,
                        "samples": len(samples), "attempts": attempts,
                        "model": args.model.name, "model_width": net.input_width,
                        "model_height": net.input_height,
                        "roi_x": state["cx"], "roi_y": state["cy"], "roi_size": 2 * args.roi_radius + 1,
                        "left_camera": args.left, "right_camera": args.right,
                        "rectify_alpha": args.alpha, "rect_focal_px": focal,
                        "rect_baseline_calib_units": baseline, "cx_offset_px": cx_offset,
                        "calibration_file": str(args.calibration),
                        "providers": ";".join(net.session.get_providers()),
                    }
                    append_csv(csv_path, row)
                    print("\n===== CREStereo: saved 50-frame series =====")
                    print(f"Real={real_cm:.2f} cm; network={avg_cm:.2f} +/- {std_cm:.3f} cm; "
                          f"absolute error={abs_error:.2f} cm; relative error={row['relative_error_percent']:.2f}%")
                    print(f"Median disparity mean={avg_d:.2f} px; positive ROI={avg_positive:.1f}% "
                          "(NOT a confidence metric)")
                    print(f"Inference={avg_inf_ms:.1f} ms ({row['algorithm_fps']:.2f} FPS); "
                          f"full cycle={avg_full_ms:.1f} ms ({row['full_cycle_fps']:.2f} FPS)")
                    print(f"CSV: {csv_path}\n")
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_snapshot(args.output, f"{stamp}_d{real_cm:g}cm", *latest_frames)
                    samples = []
                    attempts = 0
                    gathering = False
                elif attempts >= args.max_attempts:
                    print(f"Aborted: only {len(samples)} valid frames out of {attempts} attempts.")
                    gathering = False
                    samples = []
                    attempts = 0

            if key in (ord("q"), 27):
                if gathering:
                    print("Unfinished series discarded; no CSV row added.")
                break
            if key == ord("r"):
                state["cx"], state["cy"] = w // 2, h // 2
            if key == ord("s"):
                if real_cm is None:
                    print("Cannot save series without real distance. Restart with --distance 98")
                elif gathering:
                    print("A series is already in progress.")
                else:
                    print(f"Collecting {args.samples} valid measurements at real distance={real_cm:g} cm...")
                    gathering, samples, attempts = True, [], 0
            if key == ord("p") and latest_frames is not None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_snapshot(args.output, f"{stamp}_snapshot", *latest_frames)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed")
                break
    finally:
        if left_cap is not None:
            left_cap.release()
        if right_cap is not None:
            right_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
