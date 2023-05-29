import math

from src.ray_casting.point import Point
from src.ray_casting.line import Line

_eps = 0.001
_segment_equality_eps = 5


class Segment(Line):

    def min_x(self):
        return min(self.a.x, self.b.x)

    def min_y(self):
        return min(self.a.y, self.b.y)

    def max_x(self):
        return max(self.a.x, self.b.x)

    def max_y(self):
        return max(self.a.y, self.b.y)

    def center(self):
        width = abs(self.a.x - self.b.x)
        height = abs(self.a.y - self.b.y)

        min_x = min(self.a.x, self.b.x)
        min_y = min(self.a.y, self.b.y)

        return Point(min_x + width / 2, min_y + height / 2)

    def is_point_on_segment(self, pt):
        min_x = self.min_x() - _eps
        min_y = self.min_y() - _eps
        max_x = self.max_x() + _eps
        max_y = self.max_y() + _eps

        return pt.x >= min_x and pt.x <= max_x and pt.y >= min_y and pt.y <= max_y

    def intersects_with(self, segment):
        x = super().intersects_with(segment)
        if not x:
            return

        intersection_x, intersection_y = x

        return self.is_point_on_segment(Point(intersection_x, intersection_y))

    def length(self):
        a = (self.max_x() - self.min_x()) ** 2
        b = (self.max_y() - self.min_y()) ** 2

        return math.sqrt(a + b)

    def __len__(self):
        return round(self.length())

    def __eq__(self, segment2):
        return math.abs(len(self) - len(segment2)) <= _segment_equality_eps
