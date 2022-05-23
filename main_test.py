import cv2
import numpy as np
from imutils import face_utils
import dlib
import playsound
import time
import road_module as road
import eye_tracking_module as tracking
import belt_module as belt
from tkinter import *
from tkinter import messagebox

window = Tk()
window.title("Alarm")
window.geometry("300x40")
def Click():
    messagebox.showinfo("알림","뒷자석 안전벨트 해제")


cap_road = cv2.VideoCapture('video/road4.mp4')
cap_belt = cv2.VideoCapture('video/car3_2.mp4')
cap_face = cv2.VideoCapture('1.mp4')
#cap_face = cv2.VideoCapture(0)
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


detection = False
flag = True
limit = 4 # 4초


while True:
    ret_road, frame_road = cap_road.read()
    ret_face, frame_face = cap_face.read()
    ret_belt, frame_belt = cap_belt.read()
    
    frame2 = frame_road.copy()
    frame2 = cv2.resize(frame2, (1000,600))
    lineframe = frame2.copy()
    gray = cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY)
    face = detector(gray)

    if not ret_road:
        print("error")
        break

    ### line ###
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

    ### face ###
    for i in face:
        cv2.rectangle(frame_face, (i.left(), i.top()), (i.right(), i.bottom()),(0,0,255),2)
        #cv2.imshow("frame", frame_face)
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
                    playsound.playsound('audio/al2.MP3')
                    print(al_cnt1)
                else:
                    playsound.playsound('audio/al.mp3')
				
    else:
        cnt = 0
        warning = False

    ### belt ###
    origin_belt = frame_belt.copy()
    _, frame_belt2= belt.gray_img(frame_belt)
    frame_belt2 = belt.white_extract(frame_belt2)

    frame_belt2 = belt.roi_belt(frame_belt2)
    frame_belt = belt.roi_belt(frame_belt)

    gray_belt = cv2.cvtColor(frame_belt2, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_belt, cv2.HOUGH_GRADIENT, 1, 20, param1 = 250, param2 = 10, minRadius = 0, maxRadius = 50)
    
    if circles is not None:
        if circles.shape[1] == 1:
            if flag:
                begin = time.time()
                limit = begin + 10
                flag = False
        else:
            begin = time.time()
            limit = begin + 10

    if time.time() > limit:
        print('yoooooooooooo')
        #cv2.imshow('origin_belt', origin_belt)
        #cv2.waitKey(3000)
        #cv2.destroyWindow('origin_belt')
        
        bnt = Button(text="경고",command=Click)
        bnt.pack()
        window.mainloop()
        flag = True

    if circles is not None:
        for i in circles[0]:
            cv2.circle(frame_belt, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), 2)


    black = np.zeros_like(frame2)
    black = cv2.resize(black, (1000,700))
    frame_face = cv2.resize(frame_face, (500,400))
    origin_belt = cv2.resize(origin_belt, (500,400))

    black[0:300, 0:1000] = lineframe
    black[300:700, 0:500] = frame_face
    black[300:700, 500:1000] = origin_belt

    # 1번 창 화살표 추가
    # 졸음운전방지기
    cv2.putText(frame_face, "EAR:{:.2f}".format(ratio), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),3)
    #cv2.imshow("line", lineframe)
    #cv2.imshow("frame", frame_face)
    cv2.imshow("main", black)
    #cv2.imshow('frame_belt', frame_belt)


    # 버튼으로 영상의 위치를 바꿀 수 있게 만들기 
    # 버튼 누르면 blind(검정화면)
    
    if cv2.waitKey(42) == ord('q'):
        break

cap_road.release()
cv2.destroyAllWindows()