import math

import numpy as np

_eps = 0.001
_edge_equality_eps = 5


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"x: {self.x}, y: {self.y}"


class Edge:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return f"({self.a!s}), ({self.b!s})"

    def line_equasion(self):
        q = self.b.x - self.a.x
        z = self.b.y - self.a.y

        # zx - qy = z * self.a.x - q * self.a.y
        return [z, -q], z * self.a.x - q * self.a.y

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

    def point_on_edge(self, pt):
        min_x = self.min_x() - _eps
        min_y = self.min_y() - _eps
        max_x = self.max_x() + _eps
        max_y = self.max_y() + _eps

        return pt.x >= min_x and pt.x <= max_x and pt.y >= min_y and pt.y <= max_y

    def intersects_with(self, edge):
        line1 = self.line_equasion()
        line2 = edge.line_equasion()
        xy1 = line1[0]
        xy2 = line2[0]
        res1 = line1[1]
        res2 = line2[1]

        a = np.array([xy1, xy2])
        b = np.array([res1, res2])

        x = np.linalg.solve(a, b)
        # to check if true np.allclose(np.dot(a, x), b)
        intersection_x, intersection_y = x

        return self.point_on_edge(Point(intersection_x, intersection_y))

    def __len__(self):
        a = (self.max_x() - self.min_x()) ** 2
        b = (self.max_y() - self.min_y()) ** 2

        return round(math.sqrt(a + b))

    def __eq__(self, edge2):
        return math.abs(len(self) - len(edge2)) <= _edge_equality_eps


class Poly:
    def __init__(self, name, edges):
        self.name = name
        self.edges = edges

    def __str__(self):
        return f"{self.name}, {' '.join([str(edge) for edge in self.edges])}"

    def ray_goes_through(self, other_edge):
        intersected_edges = sum(edge.intersects_with(other_edge) for edge in self.edges)
        return intersected_edges >= 2


# Rect should be defined from TopLeft Point
class Rect(Poly):
    def center(self):
        x_min = min(edge.min_x() for edge in self.edges)
        x_max = max(edge.max_x() for edge in self.edges)
        y_min = min(edge.min_y() for edge in self.edges)
        y_max = max(edge.max_y() for edge in self.edges)

        width = x_max - x_min
        height = y_max - y_min

        return Point(x=x_min + width / 2, y=y_min + height / 2)
