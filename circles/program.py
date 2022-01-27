from __future__ import annotations
from enum import Enum, auto
from typing import List

class CircleTypes(Enum):
    UNDEFINED = auto()
    START = auto()
    NORMAL = auto()
    INCREMENT = auto()
    DECREMENT = auto()
    OUTPUT = auto()

class PathTypes(Enum):
    NORMAL = 0
    PRIORITY = 1
    CONDITIONAL_PRIORITY = 4
    INPUT = 2

class Circle:
    def __init__(self, index, center, radius):
        self.index = index
        self.center = center
        self.radius = radius
        self.type = CircleTypes.UNDEFINED
        self.paths = []

    def __repr__(self) -> str:
        return f"{self.type.name} Circle {self.index} <- {[p.index for p in self.paths]}"

    def connect_path(self, path:Path):
        self.paths.append(path)

class Path:
    PRIORITIES = {
        PathTypes.NORMAL: 0,
        PathTypes.PRIORITY: 1,
        PathTypes.CONDITIONAL_PRIORITY: 2,
        PathTypes.INPUT: 0
    }

    def __init__(self, index, type:PathTypes):
        self.index = index
        self.type = type
        self.circles = []

    def __repr__(self) -> str:
        return f"{self.type.name} Path {self.index} -> {self.circles}"

    def connect_circle(self, circle:Circle):
        self.circles.append(circle)
        circle.connect_path(self)

class Program:
    def __init__(self, image, circles:List[Circle], paths:List[Path]):
        self.image = image
        self.circles = circles
        self.paths = paths
        