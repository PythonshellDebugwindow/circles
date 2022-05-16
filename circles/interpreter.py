from collections import defaultdict
from enum import Enum, auto

from circles.program import Program, CircleTypes, Path
from circles.exceptions import *

class CrementModes(Enum):
    CREMENTING = auto()
    NOT_CREMENTING = auto()

class Interpreter:
    def __init__(self, program:Program) -> None:
        self.program = program

        self.step_number = -1

        self.previous:Circle = None
        self.current:Circle = None
        self.next:Circle = None

        self.last_normal_circle:Circle = None

        self.terminated = False

        self.crement_mode = CrementModes.NOT_CREMENTING
        self.crement_count = 0

    def get_start_circle(self):
        starts:List[Circle] = []
        for circle in self.program.circles:
            if circle.type == CircleTypes.START:
                starts.append(circle)
        
        if len(starts) == 0:
            raise StartCircleException("No start circle found", [])
        elif len(starts) > 1:
            raise StartCircleException("Multiple start circles found", starts)
        else:
            return starts[0]

    def start(self):
        self.current = self.get_start_circle()
        self.previous = self.current

    def interpret(self):
        self.start()

        self.step_number+=1

        while True:
            self.step()

    def step(self):
        if self.step_number<0:
            self.start()
        else:
            self.do_current_circle()
            self.go_next()
        self.step_number+=1

    def halt(self, reason:str):
        print(f"Program halted because {reason}")
        self.terminated = True

    def do_current_circle(self):
        if self.current.type == CircleTypes.START:
            if self.step_number != 0:
                self.halt("start circle reeentered")
        elif self.current.type == CircleTypes.NORMAL:
            self.last_normal_circle = self.current
            
            if self.crement_mode == CrementModes.CREMENTING:
                self.current.value += self.crement_count
                self.crement_mode = CrementModes.NOT_CREMENTING
                self.crement_count = 0
        elif self.current.type == CircleTypes.INCREMENT:
            self.crement_mode = CrementModes.CREMENTING
            self.crement_count += 1
        elif self.current.type == CircleTypes.DECREMENT:
            self.crement_mode = CrementModes.CREMENTING
            self.crement_count -= 1
        elif self.current.type == CircleTypes.OUTPUT:
            print(self.last_normal_circle.value)
    
    def go_next(self):
        next_paths = self.current.paths_that_dont_connect_to(self.previous)

        next_path_priorities = defaultdict(list)

        for next_path in next_paths:
            next_path_priority = next_path.get_priority()
            next_path_priorities[next_path_priority].append(next_path)

        max_of_next_path_priority = max(next_path_priorities.keys())

        possible_next_paths = next_path_priorities[max_of_next_path_priority]

        possible_next_paths_len = len(possible_next_paths)

        if possible_next_paths_len > 1:
            raise AmbiguousPathsException("Too many possible paths", possible_next_paths)
        elif possible_next_paths_len < 1:
            self.halt("no possible paths without going back")

        print(f"{next_path_priorities=}")
        print(f"{possible_next_paths=}")
        print(next_paths)


