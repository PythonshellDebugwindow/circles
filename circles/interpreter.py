from program import Program, CircleTypes
from exceptions import *

class Interpreter:
    def __init__(self, program:Program) -> None:
        self.program = program

    def get_start_circle(self):
        starts = []
        for circle in self.program.circles:
            if circle.type == CircleTypes.START:
                starts.append(circle)
        
        if len(starts) == 0:
            raise StartCircleException("No start circle found", [])
        elif len(starts) > 1:
            raise StartCircleException("Multiple start circles found", starts)
        else:
            return starts[0]

    def interpret(self):
        start = self.get_start_circle


