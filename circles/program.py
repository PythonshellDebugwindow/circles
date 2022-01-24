from enum import Enum, auto

class CircleTypes(Enum):
    START = auto()
    NORMAL = auto()
    INCREMENT = auto()
    DECREMENT = auto()
    OUTPUT = auto()

class Circle:
    def __init__(self, index):
        self.index = index

class Program:
    def __init__(self, circles):
        self.circles = circles
        