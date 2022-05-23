import cv2
import numpy as np

# cap_belt = cv2.VideoCapture('car3.mp4')

def gray_img(img):
    # frame = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(img, (600, 600))    
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray, frame_copy

def roi_belt(img):
    return img[270:410, 340:515]

def white_extract(img):
    bgr_threshold = [210, 210, 210]
    thresholds = (img[:,:,0] < bgr_threshold[0]) | (img[:,:,1] < bgr_threshold[1]) | (img[:,:,2] < bgr_threshold[2])
    img[thresholds] = [0,0,0]
    return img

def detect_circles(roi_img):
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 250, param2 = 10, minRadius = 0, maxRadius = 50)
    return circles