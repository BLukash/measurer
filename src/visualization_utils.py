import random

import cv2
import numpy as np


def visualize_scale_detection_results(
    image_path, detected_digits, candidate_segments, pixel_len
):
    for segment in candidate_segments:
        print(segment)
    img = cv2.imread(image_path)

    sorted_digits = sorted(detected_digits, key=lambda digit: int(digit.name))
    for idx, digit in enumerate(sorted_digits):
        xmin = round(digit.min_x())
        xmax = round(digit.max_x())
        ymin = round(digit.min_y())
        ymax = round(digit.max_y())
        cv2.putText(
            img,
            str(digit.name),
            (xmin, ymin - 15),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=2,
            color=(0, 255, 0),
            thickness=3,
        )

        color = (
            (0, 0, 255)
            if (idx == 0) or (idx == len(sorted_digits) - 1)
            else (0, 255, 0)
        )
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)

    random_segment_idx = random.randint(0, len(candidate_segments) - 1)
    segment = candidate_segments[random_segment_idx]
    color = list(np.random.random(size=3) * 256)
    segment_center = segment.center()
    cv2.putText(
        img,
        str(pixel_len),
        (round(segment_center.x - 70), round(segment_center.y) - 50),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=2,
        color=color,
        thickness=3,
    )
    cv2.line(
        img,
        (round(segment.a.x), round(segment.a.y)),
        (round(segment.b.x), round(segment.b.y)),
        color,
        6,
    )

    return img
