from collections import defaultdict
from enum import Enum, auto
import cv2

from circles.program import Program, CircleTypes, Path
from circles.exceptions import *
from circles.cv_helper import display_and_wait

class CrementModes(Enum):
    CREMENTING = auto()
    NOT_CREMENTING = auto()

class Interpreter:
    def __init__(self, program:Program, do_debug=False) -> None:
        self.program = program
        self.do_debug = do_debug

        self.step_number = -1

        self.previous:Circle = None
        self.current:Circle = None
        self.next:Circle = None

        self.last_normal_circle:Circle = None

        self.halted = False
        self.halt_reason = ""

        self.crement_mode = CrementModes.NOT_CREMENTING
        self.crement_count = 0

    def get_start_circle(self):
        starts:List[Circle] = []
        undefined_circles:List[Circle] = []
        for circle in self.program.circles:
            if circle.type == CircleTypes.START:
                starts.append(circle)
            elif circle.type == CircleTypes.UNDEFINED:
                undefined_circles.append(circle)
        
        if len(undefined_circles) > 0:
            raise UndefinedCircleException("Undefined circles found", self.program, undefined_circles)

        if len(starts) == 0:
            raise StartCircleException("No start circle found", self.program, [])
        elif len(starts) > 1:
            raise StartCircleException("Multiple start circles found", self.program, starts)
        else:
            return starts[0]

    def start(self):
        self.current = self.get_start_circle()
        self.previous = self.current

    def run(self):
        self.start()

        self.step_number+=1

        while not self.halted:
            self.step()

    def step(self):
        if self.do_debug:
            self.show_where_things_are()

        if not self.halted:
            if self.step_number<0:
                self.start()
            else:
                self.do_current_circle()
                self.go_next()
            self.step_number+=1

    def show_where_things_are(self):
        labeled_image = self.program.get_labeled_image()

        self.current.draw(labeled_image, (0,255,0))

        for path in self.current.paths_that_dont_connect_to(self.previous):
            path.draw(labeled_image)

        display_and_wait(labeled_image, "where things are")

    def halt(self, reason:str, SpecificException=HaltException):
        self.halt_reason = f"Program halted because {reason}"
        print(self.halt_reason)
        self.halted = True
        raise SpecificException(self.halt_reason, self.program, [self.current])

    def do_current_circle(self):
        if self.current.type == CircleTypes.START:
            if self.step_number != 0:
                self.halt("start circle reentered", StartReenteredException)
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

        next_paths_len = len(next_paths)

        if next_paths_len < 1:
            self.halt("there are no possible paths without going back", DeadEndException)

        next_path_priorities = defaultdict(list[Path])

        for next_path in next_paths:
            next_path_priority = next_path.get_priority()
            next_path_priorities[next_path_priority].append(next_path)

        max_of_next_path_priority = max(next_path_priorities.keys())

        possible_next_paths = next_path_priorities[max_of_next_path_priority]

        possible_next_paths_len = len(possible_next_paths)

        if possible_next_paths_len > 1:
            raise AmbiguousPathsException("Too many possible paths", self.program, [self.current], possible_next_paths)
        
        next_circle = possible_next_paths[0].connected_circle_that_is_not(self.current)
        self.previous = self.current
        self.current = next_circle