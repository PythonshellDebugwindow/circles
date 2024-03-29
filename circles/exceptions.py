from typing import List
import cv2

from circles.program import Program, Circle, Path
from circles.cv_helper import display_and_wait, put_text

class CirclesException(Exception):
    def __init__(self, message, program:Program):
        self.message = message
        self.program = program
        super().__init__(self.message)
        self.show_exception()

    def show_exception(self):
        exception_image = self.draw_exception()

        image_height, _ = exception_image.shape[:2]

        put_text(exception_image, str(type(self).__name__), (0, image_height-10), font_scale=0.7, thickness=2)

        display_and_wait(exception_image, self.message)

    def draw_exception(self):
        return self.program.get_labeled_image()

class CircleException(CirclesException):
    def __init__(self, message, program:Program, circles:List[Circle]):
        self.message = message
        self.program = program
        self.circles = circles
        super().__init__(self.message, self.program)

    def draw_exception(self):
        exception_image = self.program.get_labeled_image()

        for circle in self.circles:
            circle.draw(exception_image, color=(0, 0, 255))        

        return exception_image

class NoNormalCircleVisitedException(CircleException):
    def __init__(self, program:Program, circle:Circle):
        self.message = "No normal circle visited before entering output circle"
        self.program = program
        self.circles = [circle]
        super().__init__(self.message, self.program, [circle])

class NoNormalCircleAfterCrementationException(CircleException):
    def __init__(self, program:Program, circles:List[Circle]):
        self.message = "No normal circle after crementation"
        self.program = program
        self.circles = circles
        super().__init__(self.message, self.program, circles)

class StartCircleException(CircleException):
    pass

class UndefinedCircleException(CircleException):
    pass

class HaltException(CircleException):
    pass

class DeadEndException(HaltException):
    pass

class StartReenteredException(HaltException):
    pass

class PathException(CirclesException):
    def __init__(self, message, program:Program, paths:List[Path]):
        self.message = message
        self.program = program
        self.paths = paths
        super().__init__(self.message, self.program)

    def draw_exception(self):
        exception_image = self.program.get_labeled_image()

        for path in self.paths:
            path.draw(exception_image, color=(0, 0, 255))

        return exception_image

class CircleAndPathException(CirclesException):
    def __init__(self, message, program:Program, circles:List[Circle], paths:List[Path]):
        self.message = message
        self.program = program
        self.circles = circles
        self.paths = paths
        super().__init__(self.message, self.program)

    def draw_exception(self):
        exception_image = self.program.get_labeled_image()

        for circle in self.circles:
            circle.draw(exception_image, color=(0, 0, 255))

        for path in self.paths:
            path.draw(exception_image, color=(0, 0, 255))

        return exception_image

class AmbiguousPathsException(CircleAndPathException):
    pass