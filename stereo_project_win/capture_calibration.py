import cv2
import os


camera_left = cv2.VideoCapture(0)
camera_right = cv2.VideoCapture(1)

for camera in (camera_left, camera_right):
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    
if not camera_left.isOpened():
    print("Левая камера не открылась")
    exit()

if not camera_right.isOpened():
    print("Правая камера не открылась")
    exit()

# Папки для фотографий
os.makedirs("calibration_photos/left", exist_ok=True)
os.makedirs("calibration_photos/right", exist_ok=True)

image_number = 0

print("S - сохранить пару")
print("Q - выйти")

while True:
    # Сначала просим обе камеры захватить кадр
    camera_left.grab()
    camera_right.grab()

    # Затем получаем захваченные кадры
    ret_left, frame_left = camera_left.retrieve()
    ret_right, frame_right = camera_right.retrieve()

    if not ret_left or not ret_right:
        print("Не удалось получить кадры")
        continue

    cv2.imshow("LEFT", frame_left)
    cv2.imshow("RIGHT", frame_right)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        left_name = f"calibration_photos/left/{image_number:02d}.jpg"
        right_name = f"calibration_photos/right/{image_number:02d}.jpg"

        cv2.imwrite(left_name, frame_left)
        cv2.imwrite(right_name, frame_right)

        print(f"Сохранена пара {image_number}")

        image_number += 1

    elif key == ord("q"):
        break

camera_left.release()
camera_right.release()

cv2.destroyAllWindows()