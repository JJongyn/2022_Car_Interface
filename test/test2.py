import cv2
import dlib

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
 
while True:
    ret_val, img = cam.read() # 캠 이미지 불러오기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = detector(gray)
    for i in face:
        print(i)
        cv2.rectangle(img, (i.left(), i.top()), (i.right(), i.bottom()),(0,0,255),2)
    cv2.imshow("Cam Viewer",img) # 불러온 이미지 출력하기

    # 웹캠
    if cv2.waitKey(1) == 27:
        break  # esc to quit