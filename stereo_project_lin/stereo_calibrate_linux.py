import cv2
import numpy as np
import glob
from pathlib import Path
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 24.0  # мм

# Критерий уточнения координат углов
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

# Координаты углов шахматки в реальном мире
object_points_template = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32,
)

object_points_template[:, :2] = np.mgrid[
    0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]
].T.reshape(-1, 2)

object_points_template *= SQUARE_SIZE

# Здесь будем хранить результаты со всех фотографий
object_points = []
image_points_left = []
image_points_right = []

left_images = sorted(glob.glob("calibration_photos/left/*.jpg"))
right_images = sorted(glob.glob("calibration_photos/right/*.jpg"))

print("Левых фотографий:", len(left_images))
print("Правых фотографий:", len(right_images))

if len(left_images) != len(right_images):
    print("ОШИБКА: количество левых и правых фотографий различается")
    exit()

image_size = None

for left_name, right_name in zip(left_images, right_images):

    left = cv2.imread(left_name)
    right = cv2.imread(right_name)

    if left is None or right is None:
        print("Не удалось открыть:", left_name, right_name)
        continue

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    image_size = (gray_left.shape[1], gray_left.shape[0])

    found_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD)

    found_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD)

    if found_left and found_right:

        # Уточняем координаты углов до субпиксельной точности
        corners_left = cv2.cornerSubPix(
            gray_left,
            corners_left,
            (11, 11),
            (-1, -1),
            criteria,
        )

        corners_right = cv2.cornerSubPix(
            gray_right,
            corners_right,
            (11, 11),
            (-1, -1),
            criteria,
        )

        object_points.append(object_points_template.copy())
        image_points_left.append(corners_left)
        image_points_right.append(corners_right)

        print("OK:", left_name, right_name)

    else:
        print(
            "НЕ НАЙДЕНА:",
            left_name,
            right_name,
            "| LEFT:",
            found_left,
            "RIGHT:",
            found_right,
        )

print()
print("Подходящих стереопар:", len(object_points))

if len(object_points) < 10:
    print("Слишком мало подходящих пар для нормальной калибровки")
    exit()

# -------------------------
# Калибровка левой камеры
# -------------------------

error_left, K_left, D_left, _, _ = cv2.calibrateCamera(
    object_points,
    image_points_left,
    image_size,
    None,
    None,
)

# -------------------------
# Калибровка правой камеры
# -------------------------

error_right, K_right, D_right, _, _ = cv2.calibrateCamera(
    object_points,
    image_points_right,
    image_size,
    None,
    None,
)

print()
print("Ошибка левой камеры:", error_left)
print("Ошибка правой камеры:", error_right)

print()
print("Матрица левой камеры:")
print(K_left)

print()
print("Матрица правой камеры:")
print(K_right)

# -------------------------
# Стереокалибровка
# -------------------------

stereo_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-5,
)

stereo_error, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
    object_points,
    image_points_left,
    image_points_right,
    K_left,
    D_left,
    K_right,
    D_right,
    image_size,
    criteria=stereo_criteria,
    flags=cv2.CALIB_FIX_INTRINSIC,
)

print()
print("Ошибка стереокалибровки:", stereo_error)

print()
print("R:")
print(R)

print()
print("T:")
print(T)

# Длина вектора между оптическими центрами камер
baseline = np.linalg.norm(T)

print()
print("================================")
print("Расчётный baseline:", baseline, "мм")
print("Реальный baseline: 100 мм")
print("Ошибка baseline:", abs(baseline - 100.0), "мм")
print("================================")

# Сохраняем результат
np.savez(
    Path(__file__).resolve().parent / "stereo_calibration.npz",
    K_left=K_left,
    D_left=D_left,
    K_right=K_right,
    D_right=D_right,
    R=R,
    T=T,
    E=E,
    F=F,
    image_size=np.array(image_size),
)

print()
print("Калибровка сохранена в stereo_calibration.npz")
