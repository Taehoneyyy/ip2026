import cv2
import numpy as np


def on_change(value):
    pass


camera = cv2.VideoCapture(0)

control_window = "HSV Controller"
cv2.namedWindow(control_window)

cv2.createTrackbar("Low H", control_window, 0, 179, on_change)
cv2.createTrackbar("Low S", control_window, 0, 255, on_change)
cv2.createTrackbar("Low V", control_window, 0, 255, on_change)
cv2.createTrackbar("High H", control_window, 179, 179, on_change)
cv2.createTrackbar("High S", control_window, 255, 255, on_change)
cv2.createTrackbar("High V", control_window, 255, 255, on_change)

while True:
    success, image = camera.read()
    if not success:
        break

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    low_h = cv2.getTrackbarPos("Low H", control_window)
    low_s = cv2.getTrackbarPos("Low S", control_window)
    low_v = cv2.getTrackbarPos("Low V", control_window)
    high_h = cv2.getTrackbarPos("High H", control_window)
    high_s = cv2.getTrackbarPos("High S", control_window)
    high_v = cv2.getTrackbarPos("High V", control_window)

    lower_bound = np.array([low_h, low_s, low_v], dtype=np.uint8)
    upper_bound = np.array([high_h, high_s, high_v], dtype=np.uint8)

    color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    filtered_result = cv2.bitwise_and(image, image, mask=color_mask)

    cv2.imshow("Original", image)
    cv2.imshow("Mask", color_mask)
    cv2.imshow("Filtered", filtered_result)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()