import os
from pathlib import Path
from typing import List
import cv2
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

class Parser:
    def __init__(self, program = 5):
        self.MAX_PROGRAM = 7
        self.MIN_PROGRAM = 1
        self.program = program

    def run(self):
        self.image = cv2.imread(Parser.get_program(self.program))

        self.gray = cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)
        self.gray = cv2.bitwise_not(self.gray)
        self.gray = Parser.morph_func(self.gray, cv2.dilate)

        self.debug_out = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        circles = Parser.get_circles(self.gray, self.debug_out)

        outer_contours, _ = cv2.findContours(self.gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_contour_mask = Parser.mask_contours(outer_contours, self.gray)

        gray_distance_transform = cv2.distanceTransform(self.gray, cv2.DIST_L2, 5)
        self.half_stroke_width = np.max(gray_distance_transform)

        self.outer_distance_transform = cv2.distanceTransform(outer_contour_mask, cv2.DIST_L2, 5)
        
        self.invert_gray = cv2.bitwise_not(self.gray)
        self.area_contours, (self.retr_hierarchy,) = cv2.findContours(self.invert_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        np_area_contours = np.array(self.area_contours)

        first_found=None

        for i,c in enumerate(self.retr_hierarchy):
            if c[3]==-1:
                first_found=i

        hierarchy = {}

        found_something = False
        h_index=0

        while self.retr_hierarchy[first_found][2]!=-1:
            same_h = Parser.find_same_hierarchied(self.retr_hierarchy, first_found)

            hierarchy[h_index]=same_h

            for c in same_h:
                if self.retr_hierarchy[c][2]!=-1:
                    first_found=self.retr_hierarchy[c][2]
                    h_index+=1
                    found_something=True

        hierarchy[h_index]=Parser.find_same_hierarchied(self.retr_hierarchy, first_found)

        print(hierarchy)

        print(self.half_stroke_width)

        hole_contour_indices = [n for l in list(hierarchy.values())[2:] for n in l]
        self.hole_contours = np_area_contours[(hole_contour_indices,)]

        fill_contour_indices = [n for l in list(hierarchy.values())[:1]+list(hierarchy.values())[2:] for n in l]
        self.fill_contours = np_area_contours[(fill_contour_indices,)]

        self.gradient_moprh = Parser.morph(self.invert_gray, 2, cv2.MORPH_GRADIENT)

        rects = [cv2.boundingRect(c) for c in self.fill_contours]

        rect_kdtree = KDTree(rects)

        self.connections = defaultdict(list)

        self.holes = Parser.mask_contours(self.hole_contours, self.invert_gray)
        for i, contour in enumerate(self.fill_contours):
            c = Parser.mask_contours([self.fill_contours[i]], self.invert_gray)

            not_c = cv2.bitwise_and(cv2.bitwise_not(c), self.invert_gray)

            dilated = Parser.morph_func(c, cv2.dilate, kernel_size=int(self.half_stroke_width*4))

            dilate_and_holes = cv2.bitwise_and(not_c, dilated)

            dil_erode = Parser.morph_func(dilate_and_holes, cv2.erode, 2)

            connected_contours = self.get_filled_in(dil_erode)
            
            c_rects = [cv2.boundingRect(cc) for cc in connected_contours]

            for r in c_rects:
                q = rect_kdtree.query(r)
                if q[0]<20:
                    self.connections[i]+=[q[1]]

        show_connections = self.image.copy()

        for from_i in self.connections.keys():
            from_cnt = self.fill_contours[from_i]

            for to_i in self.connections[from_i]:
                to_cnt = self.fill_contours[to_i]

                cv2.line(show_connections, self.get_contour_centroid(from_cnt), self.get_contour_centroid(to_cnt), (255,0,0))
        cv2.imshow("display", show_connections)

    def get_filled_in(self, to_fill):
        filled_in = self.gradient_moprh.copy()

        filled_contour_intersects, _ = cv2.findContours(to_fill, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cci in filled_contour_intersects:
            cv2.floodFill(filled_in, None, (cci[0][0][0], cci[0][0][1]), 255)

        just_filled = filled_in-self.gradient_moprh
        just_filled_erode = Parser.morph_func(just_filled, cv2.erode, 2)

        filled_contours, _ = cv2.findContours(just_filled_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return filled_contours
        
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
                cv2.circle(debug, (int(c[0]), int(c[1])), int(c[2]), (0, 0, 255), 3)
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
    def morph_func(img, func, kernel_size=2):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        return func(img, kernel)

    @staticmethod
    def mask_contours(cnts, img):
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, cnts, -1, (255, 255, 255), -1, cv2.LINE_AA)
        return mask

    @staticmethod
    def get_contour_centroid(cnt):
        M = cv2.moments(cnt)
        if M['m00'] ==0:
            return (0,0)
        return (int(M['m10']/M['m00']),int(M['m01']/M['m00']))

    @staticmethod
    def find_same_hierarchied(retr_tree, index):
        next_contours = []
        next_contour=retr_tree[index][0]

        while next_contour!=-1:
            next_contours.insert(0, next_contour)
            next_contour=retr_tree[next_contour][0]

        previous_contours = []
        previous_contour = retr_tree[index][1]

        while previous_contour!=-1:
            previous_contours.append(previous_contour)
            previous_contour=retr_tree[previous_contour][1]

        return previous_contours+[index]+next_contours

    def search_circles(self, circles):
        ...
        circle_globs = cv2.cvtColor( np.zeros(self.image.shape, dtype=self.gray.dtype), cv2.COLOR_BGR2GRAY)
        ret=circle_globs.copy()
        if circles is not None:
            for c in circles[0]:
                cv2.circle(circle_globs, (int(c[0]), int(c[1])), int(self.half_stroke_width*2), 255,-1)

            circle_globs = self.morph_func( circle_globs, cv2.erode, 2)

            glob_contours, _ = cv2.findContours(circle_globs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            glob_rects = [cv2.boundingRect(g)[:2] for g in glob_contours]

            glob_rect_kdtree = KDTree(glob_rects)

            print(glob_rects)

            circle_groups = defaultdict(list)

            ret = circle_globs.copy()

            for c in circles[0]:
                q = glob_rect_kdtree.query((c[0],c[1]))
                circle_groups[q[1]].append(c)
                print(q)

            for i in circle_groups.keys():
                avg = np.average(circle_groups[i], axis=0)
                avg_int = avg.astype(np.int32)
                cv2.circle(ret, avg_int[:2], avg_int[2], 255)

            all_glob_and_holes = cv2.cvtColor(np.zeros(self.image.shape, dtype=self.invert_gray.dtype), cv2.COLOR_BGR2GRAY)

            for gc in glob_contours:
                glob_and_holes = cv2.bitwise_and(Parser.mask_contours([gc], self.invert_gray), self.holes)

                glob_and_holes_erode = self.morph_func(glob_and_holes, cv2.erode, 3)

                gahe_grad = self.morph(glob_and_holes_erode, 2, cv2.MORPH_GRADIENT)

                
                
                all_glob_and_holes=cv2.bitwise_or(all_glob_and_holes, gahe_grad)

            print(f"{circles}, {circle_groups}")
            return all_glob_and_holes

        return ret

    def loop(self):
        index=0
        while True:
            k=cv2.waitKey(0)

            if chr(k)=="q":
                exit()
            
            else:
                if chr(k)=='.':
                    index+=1
                elif chr(k)==',':
                    index-=1
                elif chr(k) in '[]':
                    if chr(k)=='[':
                        self.program-=1
                    elif chr(k)==']':
                        self.program+=1
                    self.program=np.clip(self.program, self.MIN_PROGRAM,self.MAX_PROGRAM)
                    self.run()
                else:
                    index=int(chr(k))
                index%=len(self.fill_contours)

                k = int(self.half_stroke_width)+(int(self.half_stroke_width)+1)%2
                print(f"{k=}")

                gray_blur = cv2.GaussianBlur(self.gray, (k*2+1,k*2+1),0)

                show = self.image.copy()
                circs = Parser.get_circles(gray_blur, show, min_dist=k, param1=200,param2=65)
                from_cnt = self.fill_contours[index]

                for to_i in self.connections[index]:
                    to_cnt = self.fill_contours[to_i]

                    cv2.line(show, self.get_contour_centroid(from_cnt), self.get_contour_centroid(to_cnt), (255,0,0), thickness=2)

                cv2.imshow("display", self.search_circles(circs))
            
parser = Parser(5)

parser.run()

parser.loop()