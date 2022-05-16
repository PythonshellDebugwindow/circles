import cv2
import numpy as np

def display(img, winname="display"):
    cv2.imshow(winname, img)

def display_and_wait(img, winname="display"):
    display(img, winname)
    cv2.waitKey(0)

def distance_transform(img, dist_type=cv2.DIST_L2, mask_size=0):
    return cv2.distanceTransform(img, dist_type, mask_size)

def find_contours(img, retr=cv2.RETR_TREE, approx=cv2.CHAIN_APPROX_SIMPLE):
    return cv2.findContours(img, retr, approx)

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

def morph(img, kernel_size=2, morph=cv2.MORPH_BLACKHAT):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return cv2.morphologyEx(img, morph, kernel)

def morph_func(img, func, kernel_size=2, iterations=1):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return func(img, kernel, iterations=iterations)

def mask_contours(cnts, img, color = 255):
    mask = np.zeros_like(img)
    cv2.drawContours(mask, cnts, -1, color, -1, cv2.LINE_AA)
    return mask

def get_contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00'] ==0:
        return (0,0)
    return (int(M['m10']/M['m00']),int(M['m01']/M['m00']))