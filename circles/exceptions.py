from typing import List
from program import Circle

class CirclesException(Exception):
    def __init__(self, message, circles:List[Circle]):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

class StartCircleException(CirclesException):
    pass