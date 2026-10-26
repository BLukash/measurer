from src.ray_casting.line import Line


class IntersectionSeeker:
    @staticmethod
    def get_numbers_on_line(number_1, number_2, rest_numbers):
        center_1 = number_1.center()
        center_2 = number_2.center()
        test_line = Line(center_1, center_2)

        intersected_numbers = [
            number for number in rest_numbers if number.ray_goes_through(test_line)
        ]
        return [number_1, *intersected_numbers, number_2]

    @classmethod
    def get_candidate_numbers(cls, numbers, min_segments):
        for idx, number_1 in enumerate(numbers):
            rest_numbers = numbers[:idx] + numbers[idx + 1 :]

            while rest_numbers:
                number_2 = rest_numbers.pop()

                numbers_on_line = cls.get_numbers_on_line(
                    number_1, number_2, rest_numbers
                )
                # we need at least 4 numbers on 1 line to get 3 segments, so
                if len(numbers_on_line) > min_segments:
                    return numbers_on_line

        return []
