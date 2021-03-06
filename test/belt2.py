from turtle import circle
import cv2
import numpy as np
import time

cap_belt = cv2.VideoCapture('belt_final.mp4')

while True:
    ret, frame = cap_belt.read()
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(frame, (600, 600))
    frame2 = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    bgr_threshold = [210, 210, 210]
    thresholds = (frame[:,:,0] < bgr_threshold[0]) \
                | (frame[:,:,1] < bgr_threshold[1]) \
                | (frame[:,:,2] < bgr_threshold[2])
    frame2[thresholds] = [0,0,0]

    frame2 = frame2[260:400,330:515]
    frame = frame[260:400,330:515]
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 250, param2 = 10, minRadius = 0, maxRadius = 50)
    cnt = 0
    if circles is not None:
        if circles == 1:
            cnt +=1
        for i in circles[0]:
            cv2.circle(frame, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), 2)
    cv2.imshow('circle', frame)
    if cv2.waitKey(42) == ord('q'):
        break
cap_belt.release()
cv2.destroyAllWindows()