import argparse
import cv2
from pathlib import Path

from circles.parser import Parser, DebugParser


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path", type=str, help="Path to the file to interpret")
    argparser.add_argument("-v", "--vision", action="store_true", help="Show what parser sees")
    args = argparser.parse_args()

    assert Path(args.path).is_file(), "File does not exist"
    
    parser = Parser(cv2.imread(args.path))
    parser.parse()
    parsed_program = parser.program

    print(parsed_program.circles)

    if args.vision:
        parser.display_and_wait(parser.id_debug)