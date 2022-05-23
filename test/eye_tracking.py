#!/usr/bin/python
import cv2
import numpy as np
import dlib # 68개의 점으로 얼굴을 인식한다

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:

        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
          
    cap.release()
    cv2.destroyAllWindows()
