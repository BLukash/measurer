import numpy as np

from src.number_detector.number_detector import NumberDetector
from src.ray_casting.intersection_seeker import IntersectionSeeker
from src.ray_casting.segment import Segment


class ScaleDetector:


    @staticmethod
    def get_segments_len(segments):
        return np.array([len(segment) for segment in segments])


    @staticmethod
    def get_median_segments_len(segments):
        if not len(segments):
            return

        return np.median([segment.length() for segment in segments])


    @classmethod
    def get_candidate_segments(cls, numbers):
        segments = []

        for current_number, next_number in zip(numbers, numbers[1:]):
            current_digit_name = int(current_number.name)
            next_digit_name = int(next_number.name)

            if next_digit_name == current_digit_name + 1:
                point = current_number.center()
                next_point = next_number.center()
                test_segment = Segment(point, next_point)
                segments.append(test_segment)

        if not segments:
            return []

        return segments


    @classmethod
    def get_ruler_scale(cls, candidate_segments):
        return round(cls.get_median_segments_len(candidate_segments))

    @classmethod
    def detect_ruler_scale(cls, image_path, model_type, min_segments=3):
        all_numbers = NumberDetector.get_numbers(image_path, model_type)

        # Find numbers, that lie on the one line
        numbers_on_one_line = IntersectionSeeker.get_candidate_numbers(all_numbers, min_segments=min_segments)

        # Get candidate segments based on the found numbers
        candidate_segments = cls.get_candidate_segments(numbers_on_one_line)

        # Calculate the median length of the ruler scale if any segments were found, otherwise None
        mean_len = cls.get_ruler_scale(candidate_segments) if candidate_segments else None

        return candidate_segments, mean_len, all_numbers, numbers_on_one_line
