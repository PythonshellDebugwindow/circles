from typing import List
from circles.program import Circle, Path

class CirclesException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class CircleException(CirclesException):
    def __init__(self, message, circles:List[Circle]):
        self.message = message
        super().__init__(self.message)

class StartCircleException(CircleException):
    pass

class PathException(CirclesException):
    # TODO: Add circle argument in here
    def __init__(self, message, paths:List[Path]):
        self.message = message
        super().__init__(self.message)

class AmbiguousPathsException(PathException):
    pass