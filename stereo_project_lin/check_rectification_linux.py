"""Проверка вертикального эпиполярного расхождения по шахматной доске.

Пример:
    python check_rectification_linux.py rect_left.png rect_right.png
    python check_rectification_linux.py rect_left.png rect_right.png --cols 9 --rows 6

Требуется: opencv-python, numpy. Кадры должны быть УЖЕ ректифицированы
с одной и той же калибровкой, без последующего ручного сдвига.
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


def detect_corners(gray, pattern):
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if ok:
            return np.asarray(corners, dtype=np.float32)
    ok, corners = cv2.findChessboardCorners(gray, pattern)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="Ректифицированное левое изображение")
    parser.add_argument("right", type=Path, help="Ректифицированное правое изображение")
    parser.add_argument("--cols", type=int, default=9, help="Количество внутренних углов по горизонтали")
    parser.add_argument("--rows", type=int, default=6, help="Количество внутренних углов по вертикали")
    args = parser.parse_args()

    pattern = (args.cols, args.rows)
    if args.cols < 2 or args.rows < 2:
        parser.error("Задайте положительное количество внутренних углов >= 2")
    left = cv2.imread(str(args.left))
    right = cv2.imread(str(args.right))
    if left is None or right is None:
        parser.error("Не удалось открыть одну или обе фотографии")
    if left.shape[:2] != right.shape[:2]:
        parser.error("Размеры левого и правого изображений различаются")

    l_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    l_pts = detect_corners(l_gray, pattern)
    r_pts = detect_corners(r_gray, pattern)
    if l_pts is None or r_pts is None:
        print("Не удалось найти ВСЕ внутренние углы в обоих кадрах.")
        print("Проверьте: шахматка полностью в кадре, фокус, свет, --cols/--rows.")
        print("Это не доказывает плохую калибровку: снимок может быть непригоден для теста.")
        return 2

    a = l_pts.reshape(-1, 2)
    b = r_pts.reshape(-1, 2)
    # На стереопарах с большой разницей ракурсов возможна обратная нумерация
    # углов на одном снимке; в таком случае результаты требуют ручной проверки.
    delta_y = b[:, 1] - a[:, 1]
    delta_x = a[:, 0] - b[:, 0]
    reverse_abs = np.mean(np.abs(b[::-1, 1] - a[:, 1]))
    direct_abs = np.mean(np.abs(delta_y))
    print(f"Найдено соответствующих углов: {len(a)}")
    print(f"Среднее (Y_right - Y_left): {np.mean(delta_y):+.3f} px")
    print(f"Средняя |ошибка по Y|:       {np.mean(np.abs(delta_y)):.3f} px")
    print(f"Медиана |ошибка по Y|:      {np.median(np.abs(delta_y)):.3f} px")
    print(f"RMS ошибка по Y:          {np.sqrt(np.mean(delta_y ** 2)):.3f} px")
    print(f"Максимум |ошибка по Y|:     {np.max(np.abs(delta_y)):.3f} px")
    print(f"Стандартное отклонение Y:  {np.std(delta_y):.3f} px")
    print(f"Медианный disparity (X_left - X_right): {np.median(delta_x):.2f} px")
    if reverse_abs < direct_abs * 0.5:
        print("ВНИМАНИЕ: возможно, порядок углов различается (разворот на 180 градусов).")
        print("Проверьте визуально соответствие углов; числовую оценку не используйте без проверки.")

    print("Примечание: это ошибка для этих областей кадра и этой стереопары, ")
    print("а не гарантированная оценка качества по всему полю зрения.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
