import argparse

import cv2

from src.scale_detector.scale_detector import ScaleDetector
from src.visualization_utils import visualize_scale_detection_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect 1 cm/in to pixels ratio and save an annotated image"
    )
    parser.add_argument("--input", help="Path to the image", required=True)
    parser.add_argument(
        "--output", help="Path to save the annotated image", required=True
    )
    args = parser.parse_args()

    candidate_segments, pixel_len, all_numbers, _ = ScaleDetector.detect_ruler_scale(
        args.input
    )
    if not pixel_len:
        raise SystemExit("Not able to detect ruler scale for this image")

    print(f'For image: "{args.input}": 1 cm/in - {pixel_len} px')
    annotated_image = visualize_scale_detection_results(
        args.input, all_numbers, candidate_segments, pixel_len
    )
    if not cv2.imwrite(args.output, annotated_image):
        raise SystemExit(f"Could not write {args.output}")
