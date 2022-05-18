import argparse
import cv2
from pathlib import Path

from circles.parser import Parser, DebugParser
from circles.interpreter import Interpreter
from circles.cv_helper import display_and_wait

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str, help="Path to the file to interpret")
    argparser.add_argument("-v", "--vision", action="store_true", help="show what parser sees")
    argparser.add_argument("-d", "--debug", action="store_true", help="step through the running of the program")
    args = argparser.parse_args()

    assert Path(args.path).is_file(), "File does not exist"
    
    parser = Parser(cv2.imread(args.path))
    parser.parse()
    parsed_program = parser.program

    if parsed_program is not None:
        print(parsed_program.circles)

        if args.vision:
            display_and_wait(parsed_program.get_labeled_image())

        interpreter = Interpreter(parsed_program, args.debug)
        interpreter.run()