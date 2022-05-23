
import cv2
import numpy as np

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
    #cv2.imshow('mask', img_mask)
    img_canny = cv2.Canny(img_mask, 100, 200)
    #cv2.imshow('canny', img_canny)

    # roi
    
    poly_left = np.array([[(250, 600), (450, 400), (500, 400), (500, 600)]])
    poly_right = np.array([[(500, 600), (500, 400), (600, 400), (800, 600)]])
    

    #poly_left = np.array([[(250, 600), (450, 440), (500, 440), (500, 600)]])
    #poly_right = np.array([[(500, 600), (500, 440), (600, 440), (800, 600)]])
    black_left= np.zeros_like(img_canny)
    black_right = np.zeros_like(img_canny)
    
    cv2.fillPoly(black_left, poly_left, 255)
    cv2.fillPoly(black_right, poly_right, 255)
    roi_left = cv2.bitwise_and(img_canny, black_left)
    roi_right = cv2.bitwise_and(img_canny, black_right)
    #cv2.imshow('roi_left', roi_left)
    #cv2.imshow('roi_right', roi_right)
    left_lines = cv2.HoughLinesP(roi_left, 1, np.pi/180, 50, maxLineGap=50)
    right_lines = cv2.HoughLinesP(roi_right, 1, np.pi/180, 50, maxLineGap=50)
    
    left_l = []
    right_l = []
    if left_lines is not None:
        if right_lines is not None:
            for left, right in zip(left_lines, right_lines):
                l_x1, l_y1, l_x2, l_y2 = left[0]
                r_x1, r_y1, r_x2, r_y2 = right[0]
                l_p = np.polyfit((l_x1, l_x2), (l_y1, l_y2), 1)
                r_p = np.polyfit((r_x1, r_x2), (r_y1, r_y2), 1)
                l_slope = l_p[0]
                r_slope = r_p[0]
                
                #if(slope < -1.10 or 0.8 < slope):
                #cv2.line(lineframe, (x1,y1),(x2,y2),(51,104,255),3)

                if l_slope < -0.7 and l_slope > -1.2 and r_slope > 0.7 and r_slope < 1.1: 
                    pts = np.array([[l_x1, l_y1], [r_x2, r_y2], [r_x1, r_y1], [l_x2, l_y2]])
                    #cv2.line(lineframe, (x1,y1),(x2,y2),(51,104,255),3)
                    #cv2.rectangle(lineframe, (l_x1, l_y1), (r_x2, r_y2), (255,255,0), -1)
                    #cv2.polylines(lineframe, [pts], True, (255,0,255), 2)
                    cv2.fillPoly(lineframe, [pts], (255,204,153))
            
    # 주행모드, 후진모드에 따라 영상 바뀌게 만들기?
    left_line_avg = np.average(left_l, axis=0)
    right_line_avg = np.average(right_l, axis=0)
    print(left_line_avg)

    



    lineframe = lineframe[300:600, 0:1000].copy()
    black = np.zeros_like(frame)
    black = cv2.resize(black, (1000,700))

    black[0:300, 0:1000] = lineframe

    cv2.imshow("line image", lineframe)
    
    cv2.imshow("black", black)
    
    
    
    

    #cv2.imshow('line', line_img)
    
    # cv2.imshow('org', frame2)
    if cv2.waitKey(42) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()