"""Определение индекса каждой физической камеры (OpenCV).

Закройте программы, использующие камеры. Для каждого индекса
показывается окно: закройте по очереди объективы и запишите индексы.
Пробел — следующая камера; Q или Esc — завершить проверку.
"""

import cv2


def main():
    for index in range(6):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"Индекс {index}: камера не открылась")
            cap.release()
            continue
        
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"Индекс {index}: смотрите окно; закройте объектив физической камеры")
        print("Пробел — следующая камера; Q — выход")
        failed = 0
        stop = False
        while True:
            ok, frame = cap.read()
            if not ok:
                failed += 1
                if failed >= 10:
                    print(f"Индекс {index}: невозможно получить кадры")
                    break
                continue
            failed = 0
            cv2.putText(frame, f"CAMERA INDEX: {index}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "SPACE: next | Q: quit", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Camera index check", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                break
            if key in (ord("q"), 27):
                stop = True
                break
        cap.release()
        cv2.destroyAllWindows()
        if stop:
            return


if __name__ == "__main__":
    main()
