import cv2
import numpy as np
import onnxruntime as ort

from src.constants import YOLO_SAVED_MODELS
from src.ray_casting.point import Point
from src.ray_casting.rect import Rect
from src.ray_casting.segment import Segment


class NumberDetector:
    model_max_dim = 640
    model_v7 = ort.InferenceSession(
        YOLO_SAVED_MODELS.BEST_YOLO_V7_MODEL_PATH, providers=["CPUExecutionProvider"]
    )

    @staticmethod
    def get_b_box(name, xmin, ymin, xmax, ymax):
        return Rect(
            name=name,
            edges=(
                Segment(a=Point(x=xmin, y=ymin), b=Point(x=xmax, y=ymin)),
                Segment(a=Point(x=xmax, y=ymin), b=Point(x=xmax, y=ymax)),
                Segment(a=Point(x=xmax, y=ymax), b=Point(x=xmin, y=ymax)),
                Segment(a=Point(x=xmin, y=ymax), b=Point(x=xmin, y=ymin)),
            ),
        )

    @staticmethod
    def check_two_digit_number(current_digit, next_digit, test_segment):
        width_sum = current_digit.width() + next_digit.width()
        height_sum = current_digit.height() + next_digit.height()

        return width_sum > len(test_segment) or height_sum > len(test_segment)

    @staticmethod
    def sort_numbers(numbers, ascending=True):
        return sorted(
            numbers, key=lambda number: int(number.name), reverse=(not ascending)
        )

    @classmethod
    def get_prediction_b_boxes(cls, raw_predictions, original_width, original_height):
        # Filter out empty/background predictions padding the topk-all array (batch_id >= 0)
        valid_predictions = raw_predictions[raw_predictions[:, 0] >= 0]

        detected_digits = []
        for pred in valid_predictions:
            _, xmin, ymin, xmax, ymax, class_id, score = pred

            # Parse, scale, and format the single box prediction
            box = cls.get_prediction_b_box(
                xmin, ymin, xmax, ymax, score, class_id, original_width, original_height
            )
            detected_digits.append(box)

        return detected_digits

    @classmethod
    def get_prediction_b_box(
        cls, xmin, ymin, xmax, ymax, score, class_id, original_width, original_height
    ):
        # Scale coordinates back to the original image resolution
        scaled_xmin = (xmin / cls.model_max_dim) * original_width
        scaled_xmax = (xmax / cls.model_max_dim) * original_width
        scaled_ymin = (ymin / cls.model_max_dim) * original_height
        scaled_ymax = (ymax / cls.model_max_dim) * original_height

        prediction_name = int(class_id) + 1

        return cls.get_b_box(
            prediction_name, scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax
        )

    @classmethod
    def update_two_digit_number(cls, current_digit, next_digit):
        min_x = min(current_digit.min_x(), next_digit.min_x())
        min_y = min(current_digit.min_y(), next_digit.min_y())
        max_x = max(current_digit.max_x(), next_digit.max_x())
        max_y = max(current_digit.max_y(), next_digit.max_y())
        name = int(f"{int(current_digit.name)}{int(next_digit.name)}")

        new_digit = cls.get_b_box(name, min_x, min_y, max_x, max_y)

        return new_digit

    @classmethod
    def get_digits(cls, image_path):
        img0 = cv2.imread(image_path)
        h0, w0 = img0.shape[:2]

        img = cv2.resize(img0, (cls.model_max_dim, cls.model_max_dim))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        input_name = cls.model_v7.get_inputs()[0].name
        outputs = cls.model_v7.run(None, {input_name: img})
        raw_predictions = outputs[0]

        return cls.get_prediction_b_boxes(raw_predictions, w0, h0)

    @classmethod
    def convert_digits_to_numbers(cls, digits):
        numbers = []
        idx = 0

        while idx < len(digits):
            current_digit = digits[idx]

            if idx < len(digits) - 1:  # if there is a next digit
                next_digit = digits[idx + 1]
                point = current_digit.center()
                next_point = next_digit.center()
                test_segment = Segment(point, next_point)

                if cls.check_two_digit_number(current_digit, next_digit, test_segment):
                    new_digit = cls.update_two_digit_number(current_digit, next_digit)
                    numbers.append(new_digit)
                    idx += 2  # skip the next digit as it is already used
                    continue

            numbers.append(current_digit)
            idx += 1

        return numbers

    @classmethod
    def get_numbers(cls, image_path):
        digits = cls.get_digits(image_path)
        numbers = cls.convert_digits_to_numbers(digits)

        return cls.sort_numbers(numbers)
