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
    CONDITIONAL_PRIORITY = 3
    INPUT = 2

class Circle:
    def __init__(self, index, center, radius):
        self.index = index
        self.center = center
        self.radius = radius
        self.type = CircleTypes.UNDEFINED
        self.paths:List[Path] = []
        self.value:int = 0

    def __repr__(self) -> str:
        return f"{self.type.name} Circle {self.index} <- {[p.index for p in self.paths]}"

    def connect_path(self, path:Path):
        self.paths.append(path)

    def paths_that_dont_connect_to(self, circle:Circle):
        paths:List[Path] = []
        for p in self.paths:
            if p.connected_circle_that_is_not(self).index != circle.index:
                paths.append(p)
        return paths

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
        self.circles:List[Circle] = []

    def __repr__(self) -> str:
        return f"{self.type.name} Path {self.index} -> {self.circles}"

    def connect_circle(self, circle:Circle):
        self.circles.append(circle)
        circle.connect_path(self)

    def connected_circle_that_is_not(self, circle:Circle):
        for c in self.circles:
            if c.index != circle.index:
                return c

class Program:
    def __init__(self, image, circles:List[Circle], paths:List[Path]):
        self.image = image
        self.circles = circles
        self.paths = paths
        