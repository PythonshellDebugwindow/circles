import os
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

class Parser:
    def __init__(self, program = 5):
        self.MAX_PROGRAM = 7
        self.MIN_PROGRAM = 1
        self.program = program

    @staticmethod
    def display(img):
        cv2.imshow("display", img)

    @staticmethod
    def distance_transform(img, dist_type=cv2.DIST_L2, mask_size=0):
        return cv2.distanceTransform(img, dist_type, mask_size)

    def run(self):
        self.image = cv2.imread(Parser.get_program(self.program))
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.gray_invert = cv2.bitwise_not(self.gray)

        _, self.stroke = cv2.threshold(self.gray_invert, 254, 255, cv2.THRESH_BINARY)

        _, self.fill = cv2.threshold(self.gray, 254, 255, cv2.THRESH_BINARY)

        dist = Parser.distance_transform(cv2.bitwise_not(self.fill))

        self.fill_contours, _ = Parser.find_contours(self.fill)

        fill_or_stroke = cv2.bitwise_or(self.fill, self.stroke)

        self.foreground = Parser.morph(fill_or_stroke, 3, cv2.MORPH_CLOSE) 

        self.foreground_dist_trans = Parser.distance_transform(self.foreground)

        self.foreground_dist_trans_norm = self.foreground_dist_trans/np.max(self.foreground_dist_trans)
        ret,self.foreground_dist_trans_norm_thresh = cv2.threshold(self.foreground_dist_trans_norm,0.5,1,cv2.THRESH_BINARY)

        potential_circle_contours, _ = Parser.find_contours(np.array(self.foreground_dist_trans_norm_thresh, dtype=self.fill.dtype),)
        Parser.display(dist/np.max(dist))

    @staticmethod
    def find_contours(img, retr=cv2.RETR_TREE, approx=cv2.CHAIN_APPROX_SIMPLE):
        return cv2.findContours(img, retr, approx)

    @staticmethod
    def get_program(number:int):
        return f"images\program-{number}.png"

    @staticmethod
    def get_circles(img, debug, min_dist=50, max_radius = None, param1=100, param2=50):
        if max_radius is None:
            max_radius = np.min(img.shape)

        circles = cv2.HoughCircles(
            image=img,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            maxRadius=max_radius,
        )

        if circles is not None:
            for c in circles[0]:
                cv2.circle(debug, (int(c[0]), int(c[1])), int(c[2]), (255, 0, 255), 1)
                cv2.circle(debug, (int(c[0]), int(c[1])), 2, (0, 255, 0), -1)
                pass

        return circles

    @staticmethod
    def get_lines(img, debug):
        edges = cv2.Canny(img,50,150,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/180,180)
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(debug,(x1,y1),(x2,y2),(0,0,255),2)

    @staticmethod
    def morph(img, kernel_size=2, morph=cv2.MORPH_BLACKHAT):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        return cv2.morphologyEx(img, morph, kernel)

    @staticmethod
    def morph_func(img, func, kernel_size=2, iterations=1):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        return func(img, kernel, iterations=iterations)

    @staticmethod
    def mask_contours(cnts, img, color = 255):
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, cnts, -1, color, -1, cv2.LINE_AA)
        return mask

    @staticmethod
    def get_contour_centroid(cnt):
        M = cv2.moments(cnt)
        if M['m00'] ==0:
            return (0,0)
        return (int(M['m10']/M['m00']),int(M['m01']/M['m00']))

    def loop(self):
        index=0
        mode=0
        MAX_MODE=4
        while True:
            print("=======")
            key=cv2.waitKey(0)

            if key==-1 or chr(key)=="q":
                exit()
            
            else:
                if chr(key) in ',.':
                    if chr(key)=='.':
                        index+=1
                    elif chr(key)==',':
                        index-=1
                    index %= len(self.fill_contours)
                    
                elif chr(key) in '[]':
                    if chr(key)=='[':
                        self.program-=1
                    elif chr(key)==']':
                        self.program+=1
                    self.program=np.clip(self.program, self.MIN_PROGRAM,self.MAX_PROGRAM)
                    self.run()

                elif chr(key) in "-=":
                    if chr(key)=="-":
                        mode-=1
                    elif chr(key)=="=":
                        mode+=1
                    mode%=MAX_MODE

                else:
                    continue

                if mode==0:
                    fill_contour_mask = Parser.mask_contours([self.fill_contours[index]], self.fill)

                    
                    Parser.display(fill_contour_mask)
                elif mode==1:
                    Parser.display(self.gray)
                elif mode==2:
                    debug = cv2.cvtColor(self.stroke, cv2.COLOR_GRAY2BGR) 
                    Parser.get_circles(self.stroke, debug)
                
                    Parser.display(debug)
                elif mode==3:
                    Parser.display(self.foreground_dist_trans_norm_thresh)
            print(f"{index=}")
            print(f"{key=}")
            print(f"{mode=}")


parser = Parser(6)

parser.run()

parser.loop()