from src.ray_casting.point import Point
from src.ray_casting.poly import Poly

# Rect should be defined from TopLeft Point
class Rect(Poly):
    def min_x(self):
        return min(edge.min_x() for edge in self.edges)

    def max_x(self):
        return max(edge.max_x() for edge in self.edges)

    def min_y(self):
        return min(edge.min_y() for edge in self.edges)

    def max_y(self):
        return max(edge.max_y() for edge in self.edges)

    def width(self):
        x_min = min(edge.min_x() for edge in self.edges)
        x_max = max(edge.max_x() for edge in self.edges)
        return x_max - x_min

    def height(self):
        y_min = min(edge.min_y() for edge in self.edges)
        y_max = max(edge.max_y() for edge in self.edges)
        return y_max - y_min

    def center(self):
        x_min = min(edge.min_x() for edge in self.edges)
        y_min = min(edge.min_y() for edge in self.edges)

        width = self.width()
        height = self.height()

        return Point(x=x_min + width / 2, y=y_min + height / 2)
