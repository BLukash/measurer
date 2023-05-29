class Poly:
    def __init__(self, name, edges):
        self.name = name
        self.edges = edges

    def __str__(self):
        return f"{self.name}, {' '.join([str(edge) for edge in self.edges])}"

    def ray_goes_through(self, other_edge):
        intersected_edges = sum(edge.intersects_with(other_edge) for edge in self.edges)
        return intersected_edges >= 2
