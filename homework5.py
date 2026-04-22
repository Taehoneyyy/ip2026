import cv2
import numpy as np


REFERENCE_IMAGE_PATH = "testbook.jpg"
REQUIRED_MATCHES = 10


def create_matcher():
    algorithm_type = 1
    index_options = {"algorithm": algorithm_type, "trees": 5}
    search_options = {"checks": 50}
    return cv2.FlannBasedMatcher(index_options, search_options)


def main():
    reference_image = cv2.imread(REFERENCE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        print("기준 이미지를 불러올 수 없습니다.")
        return

    detector = cv2.SIFT_create()
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference_image, None)

    if reference_descriptors is None:
        print("기준 이미지에서 특징점을 찾지 못했습니다.")
        return

    matcher = create_matcher()
    camera = cv2.VideoCapture(0)

    while True:
        success, camera_frame = camera.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = detector.detectAndCompute(gray_frame, None)

        display_image = camera_frame.copy()

        if frame_descriptors is not None and len(frame_keypoints) >= REQUIRED_MATCHES:
            knn_result = matcher.knnMatch(reference_descriptors, frame_descriptors, k=2)

            selected_matches = []
            for pair in knn_result:
                if len(pair) < 2:
                    continue

                first_match, second_match = pair
                if first_match.distance < 0.5 * second_match.distance:
                    selected_matches.append(first_match)

            match_mask = None

            if len(selected_matches) >= REQUIRED_MATCHES:
                reference_points = np.float32(
                    [reference_keypoints[m.queryIdx].pt for m in selected_matches]
                ).reshape(-1, 1, 2)

                frame_points = np.float32(
                    [frame_keypoints[m.trainIdx].pt for m in selected_matches]
                ).reshape(-1, 1, 2)

                homography, inlier_mask = cv2.findHomography(
                    reference_points,
                    frame_points,
                    cv2.RANSAC,
                    5.0
                )

                if homography is not None and inlier_mask is not None:
                    match_mask = inlier_mask.ravel().tolist()

                    ref_height, ref_width = reference_image.shape
                    corners = np.float32([
                        [0, 0],
                        [0, ref_height - 1],
                        [ref_width - 1, ref_height - 1],
                        [ref_width - 1, 0]
                    ]).reshape(-1, 1, 2)

                    projected_corners = cv2.perspectiveTransform(corners, homography)
                    display_image = cv2.polylines(
                        display_image,
                        [np.int32(projected_corners)],
                        True,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA
                    )
            else:
                print(f"매칭 수 부족: {len(selected_matches)}/{REQUIRED_MATCHES}")

            result_view = cv2.drawMatches(
                reference_image,
                reference_keypoints,
                display_image,
                frame_keypoints,
                selected_matches,
                None,
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=match_mask,
                flags=2
            )

            cv2.imshow("Feature Matching", result_view)

        if cv2.waitKey(1) != -1:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()