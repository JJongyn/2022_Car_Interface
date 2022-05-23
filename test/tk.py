from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from time import sleep
import threading
import sys

import cv2
import numpy as np
from imutils import face_utils
import dlib
import playsound
import time
import road_module as road
import eye_tracking_module as tracking
import belt_module as belt

class Ui_MainWindow(object):
    #기본적으로 창만드는 작업
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("ForSign")
        MainWindow.resize(1000, 700) #창 사이즈
        MainWindow.move(1000, 700) #창 뜰 때 위치
        #이 아래로는 나도 잘 모름 화면을 구성하고 영상을 재생하는 위젯을 만드는거 같음 유지해두는게 나을듯
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.video_viewer_label = QtWidgets.QLabel(self.centralwidget)
        self.video_viewer_label.setGeometry(QtCore.QRect(10, 10, 1000, 700))

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def Video_to_frame(self, MainWindow):

        cap_road = cv2.VideoCapture('road4.mp4') #저장된 영상 가져오기 프레임별로 계속 가져오는 듯

        ###cap으로 영상의 프레임을 가지고와서 전처리 후 화면에 띄움###
        while True:
            self.ret_road, self.frame_road = cap_road.read() #영상의 정보 저장


            frame2 = self.frame_road.copy()
            frame2 = cv2.resize(frame2, (1000,600))
            self.lineframe = frame2.copy()
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
                            cv2.fillPoly(self.lineframe, [pts], (255,204,153))

            

            if self.ret_road:
                self.rgbImage = cv2.cvtColor(self.lineframe, cv2.COLOR_BGR2RGB) #프레임에 색입히기
                self.convertToQtFormat = QImage(self.rgbImage.data, self.rgbImage.shape[1], self.rgbImage.shape[0],
                                                QImage.Format_RGB888)

                self.pixmap = QPixmap(self.convertToQtFormat)
                self.p = self.pixmap.scaled(1000, 300, QtCore.Qt.IgnoreAspectRatio) #프레임 크기 조정
            
                self.video_viewer_label.setPixmap(self.p)
                self.video_viewer_label.update() #프레임 띄우기

                sleep(0.01)  # 영상 1프레임당 0.01초로 이걸로 영상 재생속도 조절하면됨 0.02로하면 0.5배속인거임

            else:
                break

        cap_road.release()
        cv2.destroyAllWindows()

    # 창 이름 설정
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("ForSign", "ForSign"))

    # video_to_frame을 쓰레드로 사용
    #이게 영상 재생 쓰레드 돌리는거 얘를 조작하거나 함수를 생성해서 연속재생 관리해야할듯
    def video_thread(self, MainWindow):
        thread = threading.Thread(target=self.Video_to_frame, args=(self,))
        thread.daemon = True  # 프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()
#메인문
if __name__ == "__main__":
    import sys

    #화면 만들려면 기본으로 있어야 하는 코드들 건들지않기
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    #영상 스레드 시작
    ui.video_thread(MainWindow)

    #창 띄우기
    MainWindow.show()

    sys.exit(app.exec_())

