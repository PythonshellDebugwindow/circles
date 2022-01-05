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

    def run(self):
        self.image = cv2.imread(Parser.get_program(self.program))

        self.gray = cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)
        self.gray = cv2.bitwise_not(self.gray)
        self.gray = Parser.morph_func(self.gray, cv2.dilate)
        
        self.invert_gray = cv2.bitwise_not(self.gray)

        self.blank = cv2.cvtColor(np.zeros(self.image.shape, dtype=self.invert_gray.dtype), cv2.COLOR_BGR2GRAY)

        self.debug_out = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        circles = Parser.get_circles(self.gray, self.debug_out)

        outer_contours, _ = cv2.findContours(self.gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_contour_mask = Parser.mask_contours(outer_contours, self.gray)

        gray_distance_transform = cv2.distanceTransform(self.gray, cv2.DIST_L2, 5)
        self.half_stroke_width = np.max(gray_distance_transform)
        self.odd_half_stroke_width = int(self.half_stroke_width)+(int(self.half_stroke_width)+1)%2

        self.gray_blur = cv2.GaussianBlur(self.gray, (self.odd_half_stroke_width*2+1,self.odd_half_stroke_width*2+1),0)
        gray_blur_very = cv2.GaussianBlur(self.gray, (self.odd_half_stroke_width*6+1,self.odd_half_stroke_width*6+1),0)

        _, thresh = cv2.threshold(gray_blur_very, 1, 255, cv2.THRESH_BINARY)
        thresh_bordered = thresh.copy()
        print(self.image.shape)
        cv2.rectangle(thresh_bordered, (0,0), (self.image.shape[1], self.image.shape[0]), 0, 2)
        thresh_morph = Parser.morph(thresh_bordered, kernel_size=self.odd_half_stroke_width*4, morph=cv2.MORPH_ERODE)
        thresh_morph_blur = cv2.GaussianBlur(thresh_morph, (self.odd_half_stroke_width*8+1,self.odd_half_stroke_width*8+1),0)
        _, thresh_mb_thresh = cv2.threshold(thresh_morph_blur, 127, 255, cv2.THRESH_BINARY)

        self.foreground_mask = thresh_mb_thresh.copy()
        self.invert_foreground_mask = cv2.bitwise_not(self.foreground_mask)
        self.foreground_distance_transform = cv2.distanceTransform(self.foreground_mask, cv2.DIST_L2, 5)
        
        self.area_contours, (self.retr_hierarchy,) = cv2.findContours(self.invert_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.np_area_contours = np.array(self.area_contours, dtype=object)

        first_found=None

        for i,c in enumerate(self.retr_hierarchy):
            if c[3]==-1:
                first_found=i

        self.hierarchy_dict = {}

        found_something = False
        h_index=0

        while self.retr_hierarchy[first_found][2]!=-1:
            same_h = Parser.find_same_hierarchied(self.retr_hierarchy, first_found)

            self.hierarchy_dict[h_index]=same_h

            for c in same_h:
                if self.retr_hierarchy[c][2]!=-1:
                    first_found=self.retr_hierarchy[c][2]
                    h_index+=1

        self.hierarchy_dict[h_index]=Parser.find_same_hierarchied(self.retr_hierarchy, first_found)

        self.contour_dict = {}

        for h in self.hierarchy_dict.keys():
            for cont in self.hierarchy_dict[h]:
                self.contour_dict[cont]=h

        hole_contour_indices = [n for l in list(self.hierarchy_dict.values())[2::2] for n in l]
        self.hole_contours = self.np_area_contours[(hole_contour_indices,)]

        self.fill_contour_indices = [n for l in list(self.hierarchy_dict.values())[::2] for n in l]

        self.fill_contours = self.np_area_contours[(self.fill_contour_indices,)]

        self.gradient_moprh = Parser.morph(self.invert_gray, 2, cv2.MORPH_GRADIENT)

        rects = [cv2.boundingRect(c) for c in self.fill_contours]

        self.rect_kdtree = KDTree(rects)

        self.connections = defaultdict(set)

        self.all_fill_contour_hierarchies = [self.contour_dict[fci] for fci in self.fill_contour_indices]
        
        self.on_foreground_dict = defaultdict(bool)

        self.holes = Parser.mask_contours(self.hole_contours, self.invert_gray)
        for i, contour in enumerate(self.fill_contours):
            c = Parser.mask_contours([self.fill_contours[i]], self.invert_gray)

            not_c = cv2.bitwise_and(cv2.bitwise_not(c), self.invert_gray)

            from_on_foreground = self.check_if_on_foreground(c, i)

            dilated = Parser.morph_func(c, cv2.dilate, kernel_size=int(self.half_stroke_width*2+2))

            dilate_and_holes = cv2.bitwise_and(not_c, dilated)

            dil_erode = Parser.morph_func(dilate_and_holes, cv2.erode, 2)

            connected_contours = self.get_filled_in(dil_erode)
            
            c_rects = [cv2.boundingRect(cc) for cc in connected_contours]

            for r in c_rects:
                q = self.rect_kdtree.query(r)
                if q[0]<20:
                    mask_to = self.mask_contours([self.fill_contours[q[1]]], self.invert_gray)
                    to_on_foreground = self.check_if_on_foreground(mask_to, q[1])

                    self.on_foreground_dict[q[1]] = to_on_foreground
                    self.on_foreground_dict[i] = from_on_foreground

                    self.connections[i].add(q[1])
                    self.connections[q[1]].add(i)

        show_connections = self.image.copy()

        for from_i in self.connections.keys():
            from_cnt = self.fill_contours[from_i]

            for to_i in self.connections[from_i]:
                to_cnt = self.fill_contours[to_i]

                col = (255,0,0)

                if self.on_foreground_dict[to_i]:
                    col = (255,255,0)

                cv2.line(show_connections, self.get_contour_centroid(from_cnt), self.get_contour_centroid(to_cnt), col)

    def check_if_on_foreground(self,cont,cont_index):
        hierarchy = self.contour_dict[self.fill_contour_indices[cont_index]]
        below = np.zeros(self.invert_gray.shape, dtype=self.invert_gray.dtype)

        if hierarchy<np.max(self.all_fill_contour_hierarchies):
            h = hierarchy

            while h<np.max(self.all_fill_contour_hierarchies):
                h+=2
                current_h_contours = self.np_area_contours[list(self.hierarchy_dict.values())[h]]
                current_h_mask = Parser.mask_contours(current_h_contours, self.invert_gray)

                below = cv2.bitwise_or(below, current_h_mask)

        below_invert = cv2.bitwise_not(below)
        without_holes = cv2.bitwise_and(below_invert, cont)

        on_background = cv2.bitwise_and(without_holes, self.invert_foreground_mask)
        on_foreground = cv2.bitwise_and(without_holes, self.foreground_mask)

        back_sum = np.sum(on_background)
        fore_sum = np.sum(on_foreground)

        return fore_sum>back_sum

    def get_filled_in(self, to_fill):
        filled_in = self.gradient_moprh.copy()

        filled_contour_intersects, _ = cv2.findContours(to_fill, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for fci in filled_contour_intersects:
            cv2.floodFill(filled_in, None, (fci[0][0][0], fci[0][0][1]), 255)

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
        circle_globs = cv2.cvtColor( np.zeros(self.image.shape, dtype=self.gray.dtype), cv2.COLOR_BGR2GRAY)
        ret=cv2.cvtColor( circle_globs, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            for c in circles[0]:
                cv2.circle(circle_globs, (int(c[0]), int(c[1])), int(self.half_stroke_width*2), 255,-1)

            glob_contours, _ = cv2.findContours(circle_globs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            all_stuff = self.blank.copy()

            group_dict = defaultdict(list) # group:[circle, circle, circle,...],...
            circle_dict = {}               # circle:group, circle: group,...
            max_dict = {}                  # group:max, group:max,...

            for i, gc in enumerate(glob_contours):
                gc_mask = Parser.mask_contours([gc], self.invert_gray)
                gc_mask_fdt = self.foreground_distance_transform.copy()
                gc_mask_fdt[np.where(gc_mask==0)]=0
                max_gc_fdt = np.max(gc_mask_fdt)

                where_max = list(zip(*np.where(gc_mask_fdt==max_gc_fdt)))
                max_y = int(np.average([w[0] for w in where_max]))
                max_x = int(np.average([w[1] for w in where_max]))

                max_dict[i]=(max_x, max_y, max_gc_fdt)

                circle_area = self.blank.copy()

                cv2.circle(circle_area, (max_x,max_y), int(max_gc_fdt), 255,-1)

                area_and_holes = cv2.bitwise_and(circle_area, self.holes)

                area_and_holes_erode = self.morph_func(area_and_holes, cv2.erode, 4)

                filled_contour_intersects, _ = cv2.findContours(area_and_holes_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # cv2.drawContours(all_stuff, filled_contour_intersects, -1, 255, 3)

                filled_in = self.gradient_moprh.copy()

                for fci in filled_contour_intersects:
                    for fcii in fci:
                        cv2.floodFill(filled_in, None, (fcii[0][0], fcii[0][1]), 255)

                just_filled_in = filled_in-self.gradient_moprh
                just_filled_in_e = self.morph_func(just_filled_in, cv2.erode, 2)

                found_contours, _ = cv2.findContours(just_filled_in_e, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                found_bounds = [cv2.boundingRect(fc) for fc in found_contours]

                queries = [self.rect_kdtree.query(fb) for fb in found_bounds]
                for q in queries:
                    group_dict[i].append(q[1])
                    circle_dict[q[1]]=i

                all_stuff=cv2.bitwise_or(all_stuff, just_filled_in_e)

            ret = cv2.cvtColor(all_stuff, cv2.COLOR_GRAY2BGR)
            for gk in group_dict.keys():
                for gc in group_dict[gk]:
                    cv2.drawContours(ret, [self.fill_contours[gc]], -1, (0,127,255),2)
                cv2.putText(ret, str(gk), self.get_contour_centroid(self.fill_contours[group_dict[gk][0]]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,127,0), thickness=int(self.half_stroke_width))

            return ret, group_dict, circle_dict, max_dict

        return ret, None, None, None

    def loop(self):
        index=0
        cv2.imshow("display", self.image)
        while True:
            print("=======")
            key=cv2.waitKey(0)
            print(f"{index=}")

            if chr(key)=="q":
                exit()
            
            else:
                if chr(key)=='.':
                    index+=1
                elif chr(key)==',':
                    index-=1
                elif chr(key) in '[]':
                    if chr(key)=='[':
                        self.program-=1
                    elif chr(key)==']':
                        self.program+=1
                    self.program=np.clip(self.program, self.MIN_PROGRAM,self.MAX_PROGRAM)
                    self.run()
                else:
                    pass
                index%=len(self.fill_contours)

                self.odd_half_stroke_width = int(self.half_stroke_width)+(int(self.half_stroke_width)+1)%2

                show = cv2.cvtColor(self.foreground_mask, cv2.COLOR_GRAY2BGR)
                circs = Parser.get_circles(self.gray_blur, show, min_dist=self.odd_half_stroke_width, param1=200,param2=65)
                from_cnt = self.fill_contours[index]

                from_col = (0,0,255)

                if self.on_foreground_dict[index]:
                    from_col = (0,255,0)

                cv2.drawContours(show, [from_cnt], -1, from_col, 2)

                for to_i in self.connections[index]:
                    to_cnt = self.fill_contours[to_i]

                    to_col = (255,0,0)

                    if self.on_foreground_dict[to_i]:
                        to_col = (255,255,0)

                    cv2.drawContours(show, [to_cnt], -1, to_col, 2)
                    cv2.line(show, self.get_contour_centroid(from_cnt), self.get_contour_centroid(to_cnt), to_col, thickness=2)

                actual_circles = self.search_circles(circs)
                print(actual_circles[1])
                print(actual_circles[2])

                cv2.imshow("display", actual_circles[0])
                # cv2.imshow("display", show)
parser = Parser(5)

parser.run()

parser.loop()