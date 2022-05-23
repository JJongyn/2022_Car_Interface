import cv2
import numpy as np

cap_belt = cv2.VideoCapture('car3.mp4')

while True:
    ret, frame = cap_belt.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(frame, (600, 600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # blur_frame = cv2.GaussianBlur(frame, (5,5), 0)
    # canny_frame = cv2.Canny(blur_frame, 50, 150)
    bgr_threshold = [210, 210, 210]
    thresholds = (frame[:,:,0] < bgr_threshold[0]) \
                | (frame[:,:,1] < bgr_threshold[1]) \
                | (frame[:,:,2] < bgr_threshold[2])
    frame[thresholds] = [0,0,0]
    '''
    poly = np.array([[(330, 400), (330, 260), (515, 260), (515, 400)]])
    black_image = np.zeros_like(frame)
    cv2.fillPoly(black_image, poly, 255)
    roi_image = cv2.bitwise_and(frame, black_image)
    roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (5,5), 0)

    '''
    frame = frame[260:400,330:515]
    frame2 = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, imthres = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 각각의 컨투의 갯수 출력 ---⑤
    print('도형의 갯수: %d(%d)'% (len(contour), len(contour)))

    for i in contour:
        approx = cv2.approxPolyDP(i, cv2.arcLength(i,True)*0.02, True)
        (x,y,w,h) = cv2.boundingRect(i)
        pt1 = (x,y)
        pt2 = (x+w,y+h)
        #area = cv2.contoureArea(i)
        #_, radius = cv2.minEnclosingCircle(i)
        cv2.rectangle(frame2, pt1, pt2, (0,255,0),2)


    cv2.imshow('circle', frame2)

    if cv2.waitKey(42) == ord('q'):
        break
cap_belt.release()
cv2.destroyAllWindows()