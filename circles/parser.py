import cv2
import numpy as np
from scipy.spatial import KDTree

from circles.program import CircleTypes, PathTypes, Circle, Path, Program
from circles.cv_helper import *

class Parser:
    def __init__(self, image) -> None:
        self.image = image
        self.program = None

    def parse(self):
        # TODO: Make this function not smell
        FONT_SCALE = 0.7

        # Initialization
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.gray_invert = cv2.bitwise_not(self.gray)

        _, self.stroke = cv2.threshold(self.gray_invert, 150, 255, cv2.THRESH_BINARY)

        _, self.fill = cv2.threshold(self.gray, 150, 255, cv2.THRESH_BINARY)

        self.fill_contours, _ = find_contours(self.fill)

        fill_or_stroke = cv2.bitwise_or(self.fill, self.stroke)

        self.foreground = morph(fill_or_stroke, 3, cv2.MORPH_CLOSE) 

        self.foreground_dist_trans = distance_transform(self.foreground)

        # Find circles
        self.foreground_dist_trans_norm = self.foreground_dist_trans/np.max(self.foreground_dist_trans)
        ret,self.foreground_dist_trans_norm_thresh = cv2.threshold(self.foreground_dist_trans_norm,0.5,1,cv2.THRESH_BINARY)

        potential_circle_contours, _ = find_contours(np.array(self.foreground_dist_trans_norm_thresh, dtype=self.fill.dtype),)

        self.circles_debug = self.image.copy()

        hough_circles = get_hough_circles(self.stroke)

        circle_positions=hough_circles[:,:2]

        circle_pos_kdtree = KDTree(circle_positions)

        self.circles_mask = np.zeros_like(self.gray)

        self.confirmed_circles = []

        for i, pcc in enumerate(potential_circle_contours):
            pcc_mask = mask_contours([pcc], self.gray)
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

                self.confirmed_circles.append((max_x, max_y, int(max_pcc_fdt)))

        circles_kdtree = KDTree(self.confirmed_circles)

        self.circles = [Circle(i, (int(c[0]), int(c[1])), int(c[2])) for i, c in enumerate(self.confirmed_circles)]

        # Find and identify paths
        self.paths_mask = morph(cv2.subtract(self.foreground, self.circles_mask), 6, cv2.MORPH_OPEN)

        path_contours, _ = find_contours(self.paths_mask)

        circles_grad = morph(self.circles_mask, 2, cv2.MORPH_GRADIENT)

        fill_mask = self.stroke.copy()
        fill_mask=np.pad(fill_mask, (1,1), 'constant', constant_values=255)

        self.id_debug = cv2.cvtColor(self.stroke//4, cv2.COLOR_GRAY2BGR)

        self.paths = []

        stroke_widths = []

        for i, pc in enumerate(path_contours):
            pc_mask = mask_contours([pc], self.gray)
            pc_and_fill = cv2.bitwise_and(self.fill, pc_mask)
            pc_and_stroke = cv2.bitwise_and(self.stroke, pc_mask)

            pcas_distance_transform = distance_transform(pc_and_stroke)
            max_pcasdt = np.max(pcas_distance_transform)

            stroke_widths.append(max_pcasdt)
   
            pcaf_contours, _ = find_contours(pc_and_fill)
            pcafc_centroids = [get_contour_centroid(pcafc) for pcafc in pcaf_contours]
            pcafcc_avg = np.average(pcafc_centroids, axis=0)

            debuggery = self.image.copy()
            cv2.circle(debuggery, (int(pcafcc_avg[0]), int(pcafcc_avg[1])), int(max_pcasdt*2), (255,0,0), -1)

            path_center_circ = np.zeros_like(self.gray)
            cv2.circle(path_center_circ, (int(pcafcc_avg[0]), int(pcafcc_avg[1])), int(max_pcasdt*2), 255, -1)

            path_center_circ_minus_stroke = cv2.subtract(path_center_circ, self.stroke)
            pccms_contours, _ = find_contours(path_center_circ_minus_stroke)

            filled_path_center = np.zeros_like(self.gray)

            for pccmsc in pccms_contours:
                pccmsc_centroid = get_contour_centroid(pccmsc)
                cv2.floodFill(filled_path_center, fill_mask, pccmsc_centroid, 255)
            
            filled_path_center_contours, _ = find_contours(filled_path_center)

            display_and_wait(pc_and_fill)
            display_and_wait(filled_path_center)

            all_path_contours_count = len(pcaf_contours)
            path_center_contours_count = len(filled_path_center_contours)

            path_type_num = (all_path_contours_count - 2)*(int(path_center_contours_count!=all_path_contours_count))

            self.paths.append(Path(i, PathTypes(path_type_num)))

            cv2.putText(self.id_debug, PathTypes(path_type_num).name, (int(pcafcc_avg[0]), int(pcafcc_avg[1])), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,127,255), 2)

            pc_mask_dilate = morph_func(pc_mask, cv2.dilate, int(max_pcasdt*2))
            pcmd_and_circles = cv2.bitwise_and(pc_mask_dilate, self.circles_mask)
            pcmdac_contours, _ = find_contours(pcmd_and_circles)

            filled_circles_grad = circles_grad.copy()

            for pcmdacc in pcmdac_contours:
                pcmdacc_centroid = get_contour_centroid(pcmdacc)
                cv2.floodFill(filled_circles_grad, None, pcmdacc_centroid, 255)

            just_filled_circles = cv2.subtract(filled_circles_grad,circles_grad)

            jfc_contours, _ = find_contours(just_filled_circles)

            for jfcc in jfc_contours:
                jfcc_mec = cv2.minEnclosingCircle(jfcc)
                
                circle_query = circles_kdtree.query([jfcc_mec[0][0], jfcc_mec[0][1], jfcc_mec[1]])

                self.paths[i].connect_circle(self.circles[circle_query[1]])
        
        max_stroke_width = int(np.max(stroke_widths))

        # Identify circles
        for circle in self.circles:
            circle_mask = np.zeros_like(self.gray)
            cv2.circle(circle_mask, circle.center, circle.radius, 255, -1)

            circle_fill = cv2.bitwise_and(cv2.bitwise_and(self.fill, circle_mask), self.circles_mask)
            circle_fill = morph(circle_fill, 3, cv2.MORPH_OPEN)
            
            circle_center = np.zeros_like(self.gray)
            cv2.circle(circle_center, circle.center, max_stroke_width*4, 255, -1)
            circle_fill_and_center = cv2.bitwise_and(circle_fill, circle_center)
            cfac_contours, _ = find_contours(circle_fill_and_center)
            cfac_contours_count = len(cfac_contours)

            cf_contours, _ = find_contours(circle_fill)
            cf_contours_count = len(cf_contours)

            paths_count = len(circle.paths)

            if paths_count != 0:
                if cfac_contours_count==0:
                    circle.type = CircleTypes.OUTPUT
                elif cfac_contours_count==1:
                    if paths_count*2+1 == cf_contours_count:
                        circle.type = CircleTypes.NORMAL
                    elif paths_count*4+1 == cf_contours_count:
                        circle.type = CircleTypes.START
                elif cfac_contours_count==2:
                    circle.type = CircleTypes.DECREMENT
                elif cfac_contours_count==4:
                    circle.type = CircleTypes.INCREMENT
                else:
                    circle.type = CircleTypes.UNDEFINED
            else:
                circle.type = CircleTypes.UNDEFINED

            cv2.putText(self.id_debug, circle.type.name, circle.center, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255,127,0), 2)

        self.program = Program(self.image, self.circles, self.paths)
        return self.program

    

class DebugProgramParser(Parser):
    def __init__(self, program = 5):
        self.MAX_PROGRAM = 7
        self.MIN_PROGRAM = 1
        self.program_number = program
        FONT_SCALE = 0.6

    @staticmethod
    def get_program(number:int):
        return f"images\program-{number}.png"

    def parse(self):
        super().__init__(cv2.imread(self.get_program(self.program_number)))
        return super().parse()

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
                        self.program_number-=1
                    elif chr(key)==']':
                        self.program_number+=1
                    self.program_number=np.clip(self.program_number, self.MIN_PROGRAM,self.MAX_PROGRAM)

                elif chr(key) in "-=":
                    if chr(key)=="-":
                        mode-=1
                    elif chr(key)=="=":
                        mode+=1
                    mode%=MAX_MODE

                else:
                    continue

                if mode==0:
                    fill_contour_mask = mask_contours([self.fill_contours[index]], self.fill)
                    
                    display(fill_contour_mask)
                elif mode==1:
                    dst = cv2.cornerHarris(self.gray,2,3,0.04)
                    dst = cv2.dilate(dst,None)
                    disp = self.image.copy()
                    disp[dst>0.01*dst.max()]=[0,0,255]

                    display(disp)
                elif mode==2:
                    display(self.id_debug)
                elif mode==3:
                    display(self.circles_mask)
                    print(self.confirmed_circles)
            print(f"{index=}")
            print(f"{key=}")
            print(f"{mode=}")

class DebugParser(Parser):
    def __init__(self, program_path):
        self.program_path = program_path

    def parse(self):
        super().__init__(cv2.imread(self.get_program(self.program_path)))
        return super().parse()

    def loop(self):
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
                    
                elif chr(key) in "-=":
                    if chr(key)=="-":
                        mode-=1
                    elif chr(key)=="=":
                        mode+=1
                    mode%=MAX_MODE

                else:
                    continue

                if mode==0:
                    fill_contour_mask = mask_contours([self.fill_contours[index]], self.fill)
                    
                    display(fill_contour_mask)
                elif mode==1:
                    dst = cv2.cornerHarris(self.gray,2,3,0.04)
                    dst = cv2.dilate(dst,None)
                    disp = self.image.copy()
                    disp[dst>0.01*dst.max()]=[0,0,255]

                    display(disp)
                elif mode==2:
                    display(self.id_debug)
                elif mode==3:
                    display(self.circles_mask)
                    print(self.confirmed_circles)
            print(f"{key=}")
            print(f"{mode=}")