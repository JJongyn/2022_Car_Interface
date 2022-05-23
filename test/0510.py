from signal import alarm
from typing import Counter
import cv2
import numpy as np
import dlib
from imutils import face_utils
from PIL import ImageFont, ImageDraw, Image
import playsound

# 얼굴 detection x인 경우에도 경고음 발생

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


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


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
EYE_THRESH = 0.3
EYE_FRAMES = 40
cnt = 0
al_cnt1 = 0
al_cnt2 = 0
al = True
warning = False
str = "Warning!"
face_cnt = 0

while True:
	ret, frame = cap.read()
	_, result = cap.read()
		

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face = detector(gray)
	print(face)
	
	# face
	for i in face:	
		
		cv2.rectangle(frame, (i.left(), i.top()), (i.right(), i.bottom()),(0,0,255),2)
	cv2.imshow("frame", frame)
	landmarks = predictor(gray,i)
	landmarks = face_utils.shape_to_np(landmarks)
	
	# detect eyes
	left_eye = landmarks[lStart:lEnd]
	right_eye = landmarks[rStart:rEnd]
    
	# EAR 알고리즘 사용
	leftEAR = eye_ratio(left_eye)
	rightEAR = eye_ratio(right_eye)
	ratio = (leftEAR + rightEAR) / 2.0

	# 눈 그리기
	leftEyeHull = cv2.convexHull(left_eye)
	rightEyeHull = cv2.convexHull(right_eye)
	cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

	# 임계값을 기준으로 초과하면 count 해서 경고 발생
	if ratio < EYE_THRESH:
		cnt+=1
		if cnt >= EYE_FRAMES: # 설정 프레임보다 cnt가 커지면
			if not warning:
				warning = True
				print("warning!!!")
				al_cnt1 += 1
				if al_cnt1 <= 3 :
					playsound.playsound('al2.MP3')
					print(al_cnt1)
				else:
					playsound.playsound('al.mp3')
				
	else:
		cnt = 0
		warning = False

	cv2.putText(frame, "EAR:{:.2f}".format(ratio), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),3)

	cv2.imshow("frame", frame)	
	cv2.imshow("result", result)	
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
	
