
import cv2
import numpy as np

def coor(img, lines):
    s, i = lines
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-i)/s)
    x2 = int((y2-i)/s)

    p = [x1,y1,x2,y2]
    return p

cap = cv2.VideoCapture('road4.mp4')

# 흰색 추출
lower = np.array([80, 80, 80]) # 140
#upper = np.array([255, 255, 255])
upper = np.array([120, 120, 120])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("error")
        break
    
    frame2 = frame.copy()
    frame2 = cv2.resize(frame2, (1000,600))
    lineframe = frame2.copy()

    #frame = cv2.resize(frame, (1000,600))
    #frame = cv2.GaussianBlur(frame2, (5,5), 0)
    frame = cv2.bilateralFilter(frame2, -1, 10, 5)
    #frame = cv2.medianBlur(frame2,3)
    img_mask = cv2.inRange(frame, lower, upper)
    cv2.imshow('mask', img_mask)
    img_canny = cv2.Canny(img_mask, 100, 200)
    #cv2.imshow('canny', img_canny)

    # roi
    #poly = np.array([[(250, 600), (550, 480), (650, 480), (850, 600)]])
    poly = np.array([[(250, 600), (450, 400), (650, 400), (800, 600)]])
    black_image = np.zeros_like(img_canny)
    black_image2 = np.zeros_like(img_canny)
    cv2.fillPoly(black_image, poly, 255)
    roi_image = cv2.bitwise_and(img_canny, black_image)
    cv2.imshow('roi', roi_image)
    lines = cv2.HoughLinesP(roi_image, 1, np.pi/180, 50, maxLineGap=50)
    
    left_l = []
    right_l = []
    print(lines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            p = np.polyfit((x1, x2), (y1, y2), 1)
            slope = p[0]
            inter = p[1]
            
            #if(slope < -1.10 or 0.8 < slope):
            #cv2.line(lineframe, (x1,y1),(x2,y2),(51,104,255),3)

            if slope < -0.7 and slope > -1.2: 
                left_l.append((slope, inter))
                x1_left = x1
                y1_left = y1
                x2_left = x2
                y2_left = y2
                cv2.line(lineframe, (x1,y1),(x2,y2),(51,104,255),3)
                #cv2.rectangle(lineframe, (x2_left, y2_left), (x1_left+50, y1_left), (255,255,0), -1)
            elif slope > 0.7 and slope < 1.1: 
                right_l.append((slope, inter))
                x1_right = x1
                y1_right = y1
                x2_right = x2
                y2_right = y2
                cv2.line(lineframe, (x1,y1),(x2,y2),(0,255,0),3)
                #cv2.line(lineframe, (x1-50,y1),(x2,y2),(0,255,255),3)
               
    left_line_avg = np.average(left_l, axis=0)
    right_line_avg = np.average(right_l, axis=0)
    print(left_line_avg)





    cv2.imshow("black_image2", lineframe)
    
    
    
    
    

    #cv2.imshow('line', line_img)
    
    # cv2.imshow('org', frame2)
    if cv2.waitKey(42) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()