from math import sqrt
from random import random


class Point:
    """Defines a point in 2-D Euclidean space with (x, y) in [0, 1]"""

    x: float
    y: float

    def __init__(self, x: float, y: float):
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError(f"Point({x}, {y}) is not in [0, 1] range")
        self.x, self.y = x, y

    def euclid_distance(self, other: "Point") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def normalize(self, x_range: tuple[float, float], y_range: tuple[float, float]):
        if not x_range[0] <= self.x <= x_range[1] or not y_range[0] <= self.y <= y_range[1]:
            raise ValueError(
                f"Point({self.x}, {self.y}) is not in [{x_range[0]}, {x_range[1]}] x [{y_range[0]}, {y_range[1]}] range"
            )

        self.x = (self.x - x_range[0]) / (x_range[1] - x_range[0])
        self.y = (self.y - y_range[0]) / (y_range[1] - y_range[0])

    def to_tuple(self) -> tuple[float, float]:
        return self.x, self.y

    @classmethod
    def random(cls) -> "Point":
        return cls(x=random(), y=random())
