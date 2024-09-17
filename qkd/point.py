from math import sqrt
from random import random


class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError(f"Point({x}, {y}) is not in [0, 1] range")
        self.x, self.y = x, y

    def euclid_distance(self, other: "Point") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"

    @classmethod
    def random(cls) -> "Point":
        return cls(x=random(), y=random())
