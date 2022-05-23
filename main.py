import cv2
import numpy as np
from imutils import face_utils
import dlib
import playsound
import road_module as road
import eye_tracking_module as tracking



cap_road = cv2.VideoCapture('road4.mp4')

#cap_face = cv2.VideoCapture('face.mp4')
cap_face = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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
    ret_road, frame_road = cap_road.read()
    ret_face, frame_face = cap_face.read()
    frame2 = frame_road.copy()
    frame2 = cv2.resize(frame2, (1000,600))
    lineframe = frame2.copy()
    gray = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
    face = detector(gray)

    if not ret_road:
        print("error")
        break

    road_canny = road.preprocessing(frame2)
    roi_left, roi_right = road.roi(road_canny)
    
    left_lines = cv2.HoughLinesP(roi_left, 1, np.pi/180, 50, maxLineGap=50)
    right_lines = cv2.HoughLinesP(roi_right, 1, np.pi/180, 50, maxLineGap=50)

    if left_lines is not None:
        if right_lines is not None:
            for left, right in zip(left_lines, right_lines):
                l_x1, l_y1, l_x2, l_y2 = left[0]
                r_x1, r_y1, r_x2, r_y2 = right[0]
                l_p = np.polyfit((l_x1, l_x2), (l_y1, l_y2), 1)
                r_p = np.polyfit((r_x1, r_x2), (r_y1, r_y2), 1)
                if l_p[0]< -0.7 and l_p[0] > -1.2 and r_p[0] > 0.7 and r_p[0] < 1.1: # slope
                    pts = np.array([[l_x1, l_y1], [r_x2, r_y2], [r_x1, r_y1], [l_x2, l_y2]])
                    cv2.fillPoly(lineframe, [pts], (255,204,153))

    lineframe = lineframe[300:600, 0:1000].copy()

    for i in face:
        cv2.rectangle(frame_face, (i.left(), i.top()), (i.right(), i.bottom()),(0,0,255),2)
        cv2.imshow("frame", frame_face)
    landmarks = predictor(gray, i)
    landmarks = face_utils.shape_to_np(landmarks)
    left_eye, right_eye, ratio = tracking.detect_eye(landmarks)

    leftEyeHull = cv2.convexHull(left_eye)
    rightEyeHull = cv2.convexHull(right_eye)
    cv2.drawContours(frame_face, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame_face, [rightEyeHull], -1, (0, 255, 0), 1)

    # 특정 cnt이상이면 
    if ratio < EYE_THRESH:
        cnt += 1
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


    black = np.zeros_like(frame2)
    black = cv2.resize(black, (1000,700))
    frame_face = cv2.resize(frame_face, (500,400))

    black[0:300, 0:1000] = lineframe
    black[300:700, 0:500] = frame_face

    # 1번 창 화살표 추가
    # 졸음운전방지기
    cv2.putText(frame_face, "EAR:{:.2f}".format(ratio), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),3)
    cv2.imshow("line", lineframe)
    cv2.imshow("frame", frame_face)
    cv2.imshow("main", black)


    # 버튼으로 영상의 위치를 바꿀 수 있게 만들기 
    # 버튼 누르면 blind(검정화면)
    
    if cv2.waitKey(42) == ord('q'):
        break

cap_road.release()
cv2.destroyAllWindows()