import torch

from src.constants import YOLO_SAVED_MODELS
from src.ray_casting.point import Point
from src.ray_casting.segment import Segment
from src.ray_casting.rect import Rect


class NumberDetector:
    device = torch.device('cpu')
    model_v7 = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model=YOLO_SAVED_MODELS.BEST_YOLO_V7_MODEL_PATH, force_reload=False).to(device)

    @staticmethod
    def get_b_box(name, xmin, ymin, xmax, ymax):
        return Rect(name=name, edges=(
            Segment(a=Point(x=xmin, y=ymin), b=Point(x=xmax, y=ymin)),
            Segment(a=Point(x=xmax, y=ymin), b=Point(x=xmax, y=ymax)),
            Segment(a=Point(x=xmax, y=ymax), b=Point(x=xmin, y=ymax)),
            Segment(a=Point(x=xmin, y=ymax), b=Point(x=xmin, y=ymin))
        ))

    @staticmethod
    def check_two_digit_number(current_digit, next_digit, test_segment):
        width_sum = current_digit.width() + next_digit.width()
        height_sum = current_digit.height() + next_digit.height()

        return width_sum > len(test_segment) or height_sum > len(test_segment)

    @staticmethod
    def sort_numbers(numbers, ascending=True):
        return sorted(numbers, key=lambda number: int(number.name), reverse=(not ascending))

    @classmethod
    def get_prediction_b_box(cls, prediction):
        xmin, ymin, xmax, ymax, pred, name = prediction
        
        prediction_name = name + 1
        return cls.get_b_box(prediction_name, xmin, ymin, xmax, ymax)

    @classmethod
    def get_prediction_b_boxes(cls, predictions):
        return [cls.get_prediction_b_box(pred) for pred in predictions]

    @classmethod
    def update_two_digit_number(cls, current_digit, next_digit):
        min_x = min(current_digit.min_x(), next_digit.min_x())
        min_y = min(current_digit.min_y(), next_digit.min_y())
        max_x = max(current_digit.max_x(), next_digit.max_x())
        max_y = max(current_digit.max_y(), next_digit.max_y())
        name = int(f'{int(current_digit.name)}{int(next_digit.name)}')
        
        new_digit = cls.get_b_box(name, min_x, min_y, max_x, max_y)
        
        return new_digit

    @classmethod
    def get_digits(cls, image_path, model_type='v5'):
        model = cls.model_v5 if model_type == 'v5' else cls.model_v7

        detected_digits = model(image_path).xyxy[0].to(cls.device)
        return cls.get_prediction_b_boxes(detected_digits.numpy())

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
    def get_numbers(cls, image_path, model_type):
        digits = cls.get_digits(image_path, model_type)
        numbers = cls.convert_digits_to_numbers(digits)

        return cls.sort_numbers(numbers)
