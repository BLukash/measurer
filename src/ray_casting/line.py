import traceback
import numpy as np

class Line:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return f'({str(self.a)}), ({str(self.b)})'

    def equasion(self):
        q = self.b.x - self.a.x
        z = self.b.y - self.a.y

        # zx - qy = z * self.a.x - q * self.a.y
        return [z, -q], z * self.a.x - q * self.a.y

    def intersects_with(self, line):
        line1 = self.equasion()
        line2 = line.equasion()
        xy1 = line1[0]
        xy2 = line2[0]
        res1 = line1[1]
        res2 = line2[1]

        a = np.array([xy1, xy2])
        b = np.array([res1, res2])

        try:
            x = np.linalg.solve(a, b)
        except Exception:
            print(traceback.format_exc())
            return
        
        # to check if true np.allclose(np.dot(a, x), b)
        intersection_x, intersection_y = x

        return intersection_x, intersection_y
