from __future__ import annotations
from enum import Enum, auto
from typing import List
import cv2
import numpy as np


from circles.cv_helper import put_text

class CircleTypes(Enum):
    UNDEFINED = "???"
    START = "START"
    NORMAL = "NORMAL"
    INCREMENT = "+"
    DECREMENT = "-"
    OUTPUT = "OUTPUT"

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

    def draw(self, image, color=(0, 0, 255), thickness=2):
        cv2.circle(image, self.center, self.radius, color, thickness)

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

    def get_priority(self):
        return Path.PRIORITIES[self.type]

    def draw(self, image, color=(0, 255, 0), thickness=2):
        for i in range(len(self.circles)-1):
            cv2.line(image, self.circles[i].center, self.circles[i+1].center, color, thickness)

class Program:
    def __init__(self, image, circles:List[Circle], paths:List[Path]):
        self.image = image
        self.circles = circles
        self.paths = paths

    def get_labeled_image(self, font_scale=0.7):
        labeled = cv2.bitwise_not(self.image)//4

        for path in self.paths:
            path_center = np.average(np.array([c.center for c in path.circles]), axis=0)
            put_text(labeled, path.type.name, (int(path_center[0]), int(path_center[1])), color=(0,127,255))
            
        for circle in self.circles:
            put_text(labeled, circle.type.value, circle.center, color=(255,127,0))
        return labeled
        