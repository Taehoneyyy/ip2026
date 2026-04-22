import cv2
import numpy as np


def main():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        success, frame = camera.read()
        if not success:
            break

        view = frame.copy()
        target_area = view[0:500, 0:500]

        hsv_area = cv2.cvtColor(target_area, cv2.COLOR_BGR2HSV)

        green_low = np.array([40, 50, 50], dtype=np.uint8)
        green_high = np.array([80, 255, 255], dtype=np.uint8)

        green_mask = cv2.inRange(hsv_area, green_low, green_high)

        structure = np.ones((5, 5), dtype=np.uint8)
        filled_mask = cv2.morphologyEx(
            green_mask,
            cv2.MORPH_CLOSE,
            structure,
            iterations=3
        )

        contour_list, _ = cv2.findContours(
            filled_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contour_list) > 0:
            biggest = max(contour_list, key=cv2.contourArea)
            biggest_area = cv2.contourArea(biggest)

            if biggest_area > 2000:
                outer_line = cv2.convexHull(biggest)
                cv2.drawContours(target_area, [outer_line], -1, (0, 255, 0), 2)

        cv2.imshow("Mask", green_mask)
        cv2.imshow("Processed", filled_mask)
        cv2.imshow("Result", view)

        key = cv2.waitKey(1)
        if key != -1:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()