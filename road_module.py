import cv2
import numpy as np

def preprocessing(img):

    lower = np.array([80, 80, 80]) 
    upper = np.array([120, 120, 120])

    img_copy = cv2.bilateralFilter(img, -1, 10, 5)
    img_mask = cv2.inRange(img_copy, lower, upper)
    img_canny = cv2.Canny(img_mask, 100, 200)

    return img_canny

# image roi
def roi(img):
    poly_left = np.array([[(250, 600), (450, 400), (500, 400), (500, 600)]])
    poly_right = np.array([[(500, 600), (500, 400), (600, 400), (800, 600)]])
    
    black_left= np.zeros_like(img)
    black_right = np.zeros_like(img)
    
    cv2.fillPoly(black_left, poly_left, 255)
    cv2.fillPoly(black_right, poly_right, 255)
    roi_left = cv2.bitwise_and(img, black_left)
    roi_right = cv2.bitwise_and(img, black_right)

    return roi_left, roi_right

# line detect and drawing
def line_Detector(left_img, right_img):
    # left, right_image is roi_image
    left_lines = cv2.HoughLinesP(left_img, 1, np.pi/180, 50, maxLineGap=50)
    right_lines = cv2.HoughLinesP(right_img, 1, np.pi/180, 50, maxLineGap=50)

    if left_lines is not None:
        if right_lines is not None:
            for left, right in zip(left_lines, right_lines):
                l_x1, l_y1, l_x2, l_y2 = left[0]
                r_x1, r_y1, r_x2, r_y2 = right[0]
                l_p = np.polyfit((l_x1, l_x2), (l_y1, l_y2), 1)
                r_p = np.polyfit((r_x1, r_x2), (r_y1, r_y2), 1)
                l_slope = l_p[0]
                r_slope = r_p[0]

                return l_slope, r_slope, left[0], right[0]