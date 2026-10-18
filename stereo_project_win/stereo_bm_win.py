"""StereoBM на Raspberry Pi. S: 50 измерений, Q: выход.
Калибровка stereo_calibration.npz должна лежать в текущей папке.
T в калибровке должен быть в миллиметрах. Зависимости: numpy, opencv-python.
"""
import csv
import time

import cv2
import numpy as np

REAL_DISTANCE_CM = 300.0
NUM_SAMPLES = 50
LEFT_CAMERA = 0
RIGHT_CAMERA = 2
MIN_DISPARITY = -256
NUM_DISPARITIES = 256
BLOCK_SIZE = 15
ROI_RADIUS = 15
MIN_VALID_PIXELS = 50
SHOW_RIGHT = True
SHOW_DISPARITY = True
RESULTS_FILE = "bm_results_optimized.csv"


def save_experiment(samples):
    values = np.asarray(samples, dtype=np.float64)
    measured_cm = float(np.mean(values[:, 0]))
    std_cm = float(np.std(values[:, 0]))
    mean_disparity = float(np.mean(values[:, 1]))
    mean_valid = float(np.mean(values[:, 2]))
    mean_time = float(np.mean(values[:, 3]))
    absolute_error = abs(measured_cm - REAL_DISTANCE_CM)
    relative_error = absolute_error / REAL_DISTANCE_CM * 100.0
    algorithm_fps = 1.0 / mean_time if mean_time > 0 else 0.0
    print("\n========== РЕЗУЛЬТАТ ЭКСПЕРИМЕНТА ==========")
    print(f"Реальное расстояние: {REAL_DISTANCE_CM:.1f} cm")
    print(f"Измеренное расстояние: {measured_cm:.2f} cm")
    print(f"Стандартное отклонение: {std_cm:.2f} cm")
    print(f"Средняя disparity: {mean_disparity:.2f} px")
    print(f"Абсолютная ошибка: {absolute_error:.2f} cm")
    print(f"Относительная ошибка: {relative_error:.2f} %")
    print(f"Valid: {mean_valid:.1f} %")
    print(f"StereoBM time: {mean_time * 1000:.2f} ms")
    print(f"StereoBM FPS: {algorithm_fps:.1f}")
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "real_distance_cm", "measured_distance_cm", "std_cm",
                "mean_disparity_px", "absolute_error_cm",
                "relative_error_percent", "valid_percent",
                "processing_time_ms", "algorithm_fps",
            ])
        writer.writerow([
            REAL_DISTANCE_CM, measured_cm, std_cm, mean_disparity,
            absolute_error, relative_error, mean_valid,
            mean_time * 1000, algorithm_fps,
        ])
    print(f"Результат записан в {RESULTS_FILE}\n")


def main():
    if REAL_DISTANCE_CM <= 0:
        raise ValueError("REAL_DISTANCE_CM должен быть больше нуля")
    with np.load("stereo_calibration.npz") as data:
        K_left, D_left = data["K_left"], data["D_left"]
        K_right, D_right = data["K_right"], data["D_right"]
        R, T = data["R"], data["T"]
        image_size = tuple(int(v) for v in data["image_size"].ravel())
    width, height = image_size
    center_x, center_y = 320, 240
    x0, x1 = center_x - ROI_RADIUS, center_x + ROI_RADIUS + 1
    y0, y1 = center_y - ROI_RADIUS, center_y + ROI_RADIUS + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError("Область измерения выходит за границы изображения")

    # Карты рассчитываем один раз; remap использует фиксированную точку.
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1.0,
    )
    map_left_1, map_left_2 = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1, image_size, cv2.CV_16SC2,
    )
    map_right_1, map_right_2 = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2, image_size, cv2.CV_16SC2,
    )
    focal_length = float(P1[0, 0])
    baseline_mm = abs(float(P2[0, 3] / P2[0, 0]))
    depth_factor_cm = focal_length * baseline_mm / 10.0
    print("Размер изображения:", image_size)
    print(f"Фокусное расстояние: {focal_length:.2f} px")
    print(f"Baseline: {baseline_mm:.2f} мм")
    print("Principal point offset:", P1[0, 2] - P2[0, 2])

    stereo = cv2.StereoBM_create(NUM_DISPARITIES, BLOCK_SIZE)
    stereo.setMinDisparity(MIN_DISPARITY)
    stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
    stereo.setPreFilterSize(9)
    stereo.setPreFilterCap(31)
    stereo.setTextureThreshold(5)
    stereo.setUniquenessRatio(15)
    stereo.setSpeckleWindowSize(0)
    stereo.setSpeckleRange(2)
    stereo.setDisp12MaxDiff(-1)

    camera_left = cv2.VideoCapture(LEFT_CAMERA)
    camera_right = cv2.VideoCapture(RIGHT_CAMERA)
    try:
        if not camera_left.isOpened() or not camera_right.isOpened():
            raise RuntimeError("Не удалось открыть обе камеры")
        for camera in (camera_left, camera_right):
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print("\nS — собрать эксперимент; Q — выход\n")
        samples = []
        collecting = False
        fps = 0.0
        report_start = time.perf_counter()
        report_frames = 0
        capture_sum = prepare_sum = bm_sum = display_sum = 0.0
        while True:
            t0 = time.perf_counter()
            grabbed_left = camera_left.grab()
            grabbed_right = camera_right.grab()
            if not grabbed_left or not grabbed_right:
                print("Не удалось захватить кадры")
                break
            ret_left, frame_left = camera_left.retrieve()
            ret_right, frame_right = camera_right.retrieve()
            if not ret_left or not ret_right:
                print("Не удалось получить кадры")
                break
            t1 = time.perf_counter()
            if (frame_left.shape[1::-1] != image_size
                    or frame_right.shape[1::-1] != image_size):
                raise RuntimeError(
                    f"Размер кадров не совпадает с калибровкой: "
                    f"L={frame_left.shape[1::-1]}, R={frame_right.shape[1::-1]}, "
                    f"калибровка={image_size}"
                )
            # Ректифицируем только один канал.
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            gray_left = cv2.remap(
                gray_left, map_left_1, map_left_2, cv2.INTER_LINEAR,
            )
            gray_right = cv2.remap(
                gray_right, map_right_1, map_right_2, cv2.INTER_LINEAR,
            )
            
            # Исправляем смещение principal point в правой камере, чтобы оно совпадало с левой.

            #-----------------------------------
            # Вычисляем disparity и расстояние
            t2 = time.perf_counter()
            disparity_raw = stereo.compute(gray_left, gray_right)
            t3 = time.perf_counter()
            processing_time = t3 - t2
            # Переводим в float только область измерения.
            disparity_roi = disparity_raw[y0:y1, x0:x1].astype(np.float32) / 16.0
            
            # Используем этот блок только при положительной диспаратности
            # valid_disparities = disparity_roi[
            #     (disparity_roi > max(1, MIN_DISPARITY)) &
            #     (disparity_roi < MIN_DISPARITY + NUM_DISPARITIES)
            # ]

            # Этот блок используем при отрицательной диспаратности
            valid_disparities = disparity_roi[
                (disparity_roi > MIN_DISPARITY - 1) 
                & (disparity_roi < MIN_DISPARITY + NUM_DISPARITIES) 
                & (np.abs(disparity_roi) > 1.0)
            ]
            valid_count = valid_disparities.size
            valid_percent = 100.0 * valid_count / disparity_roi.size
            distance_cm = None
            median_disparity = None
            if valid_count >= MIN_VALID_PIXELS:
                median_disparity = float(np.median(valid_disparities))
                distance_cm = depth_factor_cm / abs(median_disparity)
                if collecting:
                    samples.append((distance_cm, median_disparity,
                                    valid_percent, processing_time))
            left_view = cv2.cvtColor(gray_left, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(left_view, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
            cv2.putText(left_view, f"FPS: {fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(left_view, f"Valid: {valid_percent:.1f}%", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if distance_cm is not None:
                cv2.putText(left_view, f"{distance_cm:.1f} cm",
                            (center_x + 25, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                cv2.putText(left_view, f"d = {median_disparity:.1f} px",
                            (center_x + 25, center_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(left_view, "NO DATA", (center_x + 25, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if collecting:
                cv2.putText(left_view, f"COLLECTING: {len(samples)}/{NUM_SAMPLES}",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
            cv2.imshow("LEFT", left_view)
            if SHOW_RIGHT:
                cv2.imshow("RIGHT", gray_right)
            if SHOW_DISPARITY:
                disparity_view = cv2.normalize(
                    disparity_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U,
                )
                cv2.imshow("DISPARITY", disparity_view)
            key = cv2.waitKey(1) & 0xFF
            t4 = time.perf_counter()
            report_frames += 1
            capture_sum += t1 - t0
            prepare_sum += t2 - t1
            bm_sum += t3 - t2
            display_sum += t4 - t3
            elapsed = t4 - report_start
            if elapsed >= 1.0:
                fps = report_frames / elapsed
                ms_per_frame = 1000.0 / report_frames
                print(
                    f"FPS: {fps:.1f} | "
                    f"Захват: {capture_sum * ms_per_frame:.1f} мс | "
                    f"Подготовка: {prepare_sum * ms_per_frame:.1f} мс | "
                    f"BM: {bm_sum * ms_per_frame:.1f} мс | "
                    f"ROI/вывод: {display_sum * ms_per_frame:.1f} мс"
                )
                report_frames = 0
                capture_sum = prepare_sum = bm_sum = display_sum = 0.0
                report_start = t4
            if collecting and len(samples) >= NUM_SAMPLES:
                save_experiment(samples)
                samples.clear()
                collecting = False
                # Исключаем сохранение CSV из следующего интервала FPS.
                report_start = time.perf_counter()
                report_frames = 0
                capture_sum = prepare_sum = bm_sum = display_sum = 0.0
            if key == ord("q"):
                break
            if key == ord("s") and not collecting:
                samples.clear()
                collecting = True
                print("Начат сбор 50 валидных измерений")
    finally:
        camera_left.release()
        camera_right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
