import os
from pathlib import Path as FilePath
from textwrap import fill
import cv2
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

from program import PathTypes, Circle, Path
class Parser:
    def __init__(self, program = 5):
        self.MAX_PROGRAM = 7
        self.MIN_PROGRAM = 1
        self.program = program

    @staticmethod
    def display(img):
        cv2.imshow("display", img)

    @staticmethod
    def display_and_wait(img):
        Parser.display(img)
        cv2.waitKey(0)

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

        self.circles_debug = self.image.copy()

        hough_circles = Parser.get_hough_circles(self.stroke)

        circle_positions=hough_circles[:,:2]

        circle_pos_kdtree = KDTree(circle_positions)

        self.circles_mask = np.zeros_like(self.gray)

        self.confirmed_circles = []

        for i, pcc in enumerate(potential_circle_contours):
            pcc_mask = Parser.mask_contours([pcc], self.gray)
            pcc_mask_fdt = self.foreground_dist_trans.copy()
            pcc_mask_fdt[np.where(pcc_mask==0)]=0
            max_pcc_fdt = np.max(pcc_mask_fdt)

            where_max = list(zip(*np.where(pcc_mask_fdt==max_pcc_fdt)))
            max_x = int(np.average([w[1] for w in where_max]))
            max_y = int(np.average([w[0] for w in where_max]))

            query=circle_pos_kdtree.query((max_x, max_y))

            if query[0]<max_pcc_fdt:
                cv2.circle(self.circles_debug, (max_x, max_y), int(max_pcc_fdt), (255,0,0), -1)
                cv2.circle(self.circles_mask, (max_x, max_y), int(max_pcc_fdt), 255, -1)

                self.confirmed_circles.append(hough_circles[query[1]])

        circles_kdtree = KDTree(self.confirmed_circles)

        self.circles = [Circle(i, (int(c[0]), int(c[1])), int(c[2])) for i, c in enumerate(self.confirmed_circles)]

        self.paths_mask = Parser.morph(cv2.subtract(self.foreground, self.circles_mask), 6, cv2.MORPH_OPEN)

        path_contours, _ = Parser.find_contours(self.paths_mask)

        circles_grad = Parser.morph(self.circles_mask, 2, cv2.MORPH_GRADIENT)

        fill_mask = self.stroke.copy()
        fill_mask=np.pad(fill_mask, (1,1), 'constant', constant_values=255)

        self.paths_debug = self.image.copy()

        self.paths = []

        stroke_widths = []

        for i, pc in enumerate(path_contours):
            pc_mask = Parser.mask_contours([pc], self.gray)
            pc_and_fill = cv2.bitwise_and(self.fill, pc_mask)
            pc_and_stroke = cv2.bitwise_and(self.stroke, pc_mask)

            pcas_distance_transform = Parser.distance_transform(pc_and_stroke)
            max_pcasdt = np.max(pcas_distance_transform)

            stroke_widths.append(max_pcasdt)
   
            pcaf_contours, _ = Parser.find_contours(pc_and_fill)
            pcafc_centroids = [Parser.get_contour_centroid(pcafc) for pcafc in pcaf_contours]
            pcafcc_avg = np.average(pcafc_centroids, axis=0)

            debuggery = self.image.copy()
            cv2.circle(debuggery, (int(pcafcc_avg[0]), int(pcafcc_avg[1])), int(max_pcasdt*2), (255,0,0), -1)

            path_center_circ = np.zeros_like(self.gray)
            cv2.circle(path_center_circ, (int(pcafcc_avg[0]), int(pcafcc_avg[1])), int(max_pcasdt*2), 255, -1)

            path_center_circ_minus_stroke = cv2.subtract(path_center_circ, self.stroke)
            pccms_contours, _ = Parser.find_contours(path_center_circ_minus_stroke)

            filled_path_center = np.zeros_like(self.gray)

            for pccmsc in pccms_contours:
                pccmsc_centroid = Parser.get_contour_centroid(pccmsc)
                cv2.floodFill(filled_path_center, fill_mask, pccmsc_centroid, 255)
            
            filled_path_center_contours, _ = Parser.find_contours(filled_path_center)

            all_path_contours_count = len(pcaf_contours)
            path_center_contours_count = len(filled_path_center_contours)

            path_type_num = (all_path_contours_count - 2)*(int(path_center_contours_count!=all_path_contours_count))

            self.paths.append(Path(i, PathTypes(path_type_num)))

            cv2.putText(self.paths_debug, PathTypes(path_type_num).name, (int(pcafcc_avg[0]), int(pcafcc_avg[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,127,255), 2)

            pc_mask_dilate = Parser.morph_func(pc_mask, cv2.dilate, int(max_pcasdt*2))
            pcmd_and_circles = cv2.bitwise_and(pc_mask_dilate, self.circles_mask)
            pcmdac_contours, _ = Parser.find_contours(pcmd_and_circles)

            filled_circles_grad = circles_grad.copy()

            for pcmdacc in pcmdac_contours:
                pcmdacc_centroid = Parser.get_contour_centroid(pcmdacc)
                cv2.floodFill(filled_circles_grad, None, pcmdacc_centroid, 255)

            just_filled_circles = cv2.subtract(filled_circles_grad,circles_grad)

            jfc_contours, _ = Parser.find_contours(just_filled_circles)

            for jfcc in jfc_contours:
                jfcc_mec = cv2.minEnclosingCircle(jfcc)
                
                circle_query = circles_kdtree.query([jfcc_mec[0][0], jfcc_mec[0][1], jfcc_mec[1]])

                self.paths[i].connect_circle(self.circles[circle_query[1]])
        
        print(self.paths)
        Parser.display(self.paths_debug)

    @staticmethod
    def find_contours(img, retr=cv2.RETR_TREE, approx=cv2.CHAIN_APPROX_SIMPLE):
        return cv2.findContours(img, retr, approx)

    @staticmethod
    def get_program(number:int):
        return f"images\program-{number}.png"

    @staticmethod
    def get_hough_circles(img, debug=None, min_dist=50, max_radius = None, param1=100, param2=50):
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
            if debug is not None:
                for c in circles[0]:
                    cv2.circle(debug, (int(c[0]), int(c[1])), int(c[2]), (255, 0, 255), 1)
                    cv2.circle(debug, (int(c[0]), int(c[1])), 2, (0, 255, 0), -1)

            return circles[0]

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
        mask = np.zeros_like(img)
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
                    dst = cv2.cornerHarris(self.gray,2,3,0.04)
                    dst = cv2.dilate(dst,None)
                    disp = self.image.copy()
                    disp[dst>0.01*dst.max()]=[0,0,255]

                    Parser.display(disp)
                elif mode==2:
                    Parser.display(self.paths_debug)
                elif mode==3:
                    Parser.display(self.circles_mask)
                    print(self.confirmed_circles)
            print(f"{index=}")
            print(f"{key=}")
            print(f"{mode=}")


parser = Parser(6)

parser.run()

parser.loop()