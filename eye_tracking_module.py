import cv2
import numpy as np
from signal import alarm
from typing import Counter
import cv2
import numpy as np
import dlib
from imutils import face_utils
from PIL import ImageFont, ImageDraw, Image
import playsound


# 유클리디안 거리
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# 유클리디안 거리(눈 비율 측정)
def eye_ratio(eye):
	# EAR 알고리즘을 통해 비율 계산
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_eye(landmarks):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    left_eye = landmarks[lStart:lEnd]
    right_eye = landmarks[rStart:rEnd]
    
	# EAR 알고리즘 사용
    leftEAR = eye_ratio(left_eye)
    rightEAR = eye_ratio(right_eye)
    ratio = (leftEAR + rightEAR) / 2.0

    return left_eye, right_eye, ratio