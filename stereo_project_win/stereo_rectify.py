import cv2
import numpy as np
import time
# Загружаем результаты нашей калибровки
data = np.load("stereo_calibration.npz")

K_left = data["K_left"]
D_left = data["D_left"]

K_right = data["K_right"]
D_right = data["D_right"]

R = data["R"]
T = data["T"]

image_size = tuple(data["image_size"].astype(int))

print("Размер изображения:", image_size)
print("T:")
print(T)

# -----------------------------------
# Вычисляем параметры ректификации
# -----------------------------------

R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
    K_left,
    D_left,
    K_right,
    D_right,
    image_size,
    R,
    T,
    alpha=1.0,
)

# -----------------------------------
# Создаём карты преобразования
# -----------------------------------

map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    K_left,
    D_left,
    R1,
    P1,
    image_size,
    cv2.CV_32FC1,
)

map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    K_right,
    D_right,
    R2,
    P2,
    image_size,
    cv2.CV_32FC1,
)

# -----------------------------------
# Открываем камеры
# -----------------------------------

camera_left = cv2.VideoCapture(2)
camera_right = cv2.VideoCapture(0)

# ============================================================
# ПРИНУДИТЕЛЬНО ИСПОЛЬЗУЕМ ИНТЕРФЕЙС MJPG ДЛЯ ФИКСА ЗЕЛЕНЫХ ЗОН
# ============================================================
for camera in (camera_left, camera_right):
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


if not camera_left.isOpened():
    print("Левая камера не открылась")
    exit()

if not camera_right.isOpened():
    print("Правая камера не открылась")
    exit()

print("Q - выход")


fps_start = time.perf_counter()
fps_frames = 0
fps = 0.0
while True:

    # Стараемся получить кадры максимально близко по времени
    camera_left.grab()
    camera_right.grab()

    ret_left, frame_left = camera_left.retrieve()
    ret_right, frame_right = camera_right.retrieve()

    if not ret_left or not ret_right:
        print("Не удалось получить кадры")
        continue

    # -----------------------------------
    # Ректификация
    # -----------------------------------

    rectified_left = cv2.remap(
        frame_left,
        map_left_x,
        map_left_y,
        cv2.INTER_LINEAR,
    )

    rectified_right = cv2.remap(
        frame_right,
        map_right_x,
        map_right_y,
        cv2.INTER_LINEAR,
    )

    # -----------------------------------
    # Склеиваем изображения
    # -----------------------------------

    stereo = np.hstack((rectified_left, rectified_right))


    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imwrite("rect_left.png", rectified_left)
        cv2.imwrite("rect_right.png", rectified_right)
        print("Стереопара сохранена")

    if key == ord("q"):
        break
    # -----------------------------------
    # Рисуем горизонтальные линии
    # -----------------------------------

    height = stereo.shape[0]
    width = stereo.shape[1]

    for y in range(0, height, 40):
        cv2.line(
            stereo,
            (0, y),
            (width, y),
            (0, 255, 0),
            1,
        )
    fps_frames += 1
    elapsed = time.perf_counter() - fps_start

    if elapsed >= 0.5:
        fps = fps_frames / elapsed
        fps_frames = 0
        fps_start = time.perf_counter()


        cv2.putText(stereo, f"FPS: {fps:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("RECTIFIED STEREO", stereo)


camera_left.release()
camera_right.release()

cv2.destroyAllWindows()
