
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

cap = cv2.VideoCapture('road2.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    frame2 = frame.copy()
    frame2 = cv2.resize(frame2, (1000,600))
    if not ret:
        print("error")
        break
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    blur_frame2 = cv2.GaussianBlur(frame2, (5,5), 0)
    canny_frame2 = cv2.Canny(blur_frame2, 50, 150)
    #cv2.imshow('canny', canny_frame2)
    
    # roi
    poly = np.array([[(100, 550), (450, 350), (650, 350), (950, 550)]])
    black_image = np.zeros_like(canny_frame2)
    cv2.fillPoly(black_image, poly, 255)
    roi_image = cv2.bitwise_and(canny_frame2, black_image)
    cv2.imshow('roi', roi_image)
    
    # line
    lines = cv2.HoughLinesP(roi_image, 1, np.pi/180, 100, minLineLength=10, maxLineGap=5)
    left_line = []
    right_line = []
    line_img = np.zeros_like(frame2)

    for line in lines:
        
        x1, y1, x2, y2 = line.reshape(4)
        p = np.polyfit((x1, x2), (y1, y2), 1)
        slope = p[0]
        inter = p[1]
        if(slope < -1.10 or 0.8 < slope):
            cv2.line(line_img, (x2,y2),(x1,y1),(255,0,0),5)

        if slope < 0: left_line.append((slope, inter))
        else: right_line.append((slope, inter))
    
    left_line_avg = np.average(left_line, axis=0)
    right_line_avg = np.average(right_line, axis=0)

    '''
    print(left_line_avg[0])
    left = coor(roi_image, left_line_avg)
    right = coor(roi_image, right_line_avg)
    
    all_lines = [left, right]

    line_img = np.zeros_like(frame2)
    for line in all_lines:
        try:
            x1,y1,x2,y2 = line
        except:
            x1,y1,x2,y2 = line[0]
        cv2.line(line_img, (x2,y2),(x1,y1),(255,0,0),10)
    '''            
    cv2.imshow('line', line_img)
    
    cv2.imshow('org', frame2)
    if cv2.waitKey(42) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()