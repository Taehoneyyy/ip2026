import cv2

image_original = cv2.imread("mountain.jpg")
image_view = image_original.copy()

window_name = "image"
cv2.namedWindow(window_name)

start_x, start_y = -1, -1
end_x, end_y = -1, -1
is_dragging = False


def on_trackbar(value):
    pass


cv2.createTrackbar("value", window_name, 0, 255, on_trackbar)


def mouse_event(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, is_dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        end_x, end_y = x, y
        is_dragging = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging:
            end_x, end_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        end_x, end_y = x, y
        is_dragging = False


cv2.setMouseCallback(window_name, mouse_event)

text_font = cv2.FONT_HERSHEY_SIMPLEX
blend_ratio = 0.3

while True:
    image_view = image_original.copy()
    track_value = cv2.getTrackbarPos("value", window_name)

    if start_x != -1 and start_y != -1 and end_x != -1 and end_y != -1:
        if is_dragging:
            temp = image_view.copy()
            cv2.rectangle(temp, (start_x, start_y), (end_x, end_y), (0, 0, 255), -1)
            image_view = cv2.addWeighted(temp, blend_ratio, image_view, 1 - blend_ratio, 0)

    info_text = f"Mouse position : ({start_x}, {start_y}) - ({end_x}, {end_y}) - {track_value}"
    cv2.putText(
        image_view,
        info_text,
        (10, 30),
        text_font,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow(window_name, image_view)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()