import threading

import numpy as np
import cv2
import dlib
import math
import time

imagePath = "steph.jpg"
FACECASCADE_PATH = "haarcascade_frontalface_default.xml"
CUSTOMCASCADE_PATH = "cascade2000_30.xml"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


faceCascade = cv2.CascadeClassifier(FACECASCADE_PATH)
customCascade = cv2.CascadeClassifier(CUSTOMCASCADE_PATH);
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# 얼굴의 각 구역의 포인트들을 구분해 놓기
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

detectrunning = 0
userValid = 0
"""
def userDetect(gray, frame):
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    redPoint = customCascade.detectMultiScale(gray)
    print("s")
    for (x, y, w, h) in faces:
        for (px, py, pw, ph) in redPoint:
            # check faces has the red point
            # put the eye pupil dectection in here
            if x <= px and px + pw <= x + w and y <= py and py + ph <= y + h:
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
                userValid = 1
            else :
                userValid = 0

            threading.Timer(1.0, userDetect,[gray,frame]).start()
"""

""" 
    def = dlib를 이용 얼굴과 눈을 찾는 함수
    input = 그레이 스케일 이미지
    output = 얼굴 중요 68개의 포인트 에 그려진 점 + 이미지
"""
def detect(gray,frame):
    # 일단, 등록한 Cascade classifier 를 이용 얼굴을 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    redPoint = customCascade.detectMultiScale(gray)

    numerator = 0
    denominator = 0

    # 얼굴에서 랜드마크를 찾자
    for (x, y, w, h) in faces:
        for(px,py,pw,ph) in redPoint:
            # check faces has the red point
            # put the eye pupil dectection in here
            if x <= px and px + pw <= x + w and y <= py and py + ph <= y + h:

                # 얼굴: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 255 0 0 , 굵기 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

                # 오픈 CV 이미지를 dlib용 사각형으로 변환하고
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

                # 랜드마크 포인트들 지정
                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
                landmarks_display = landmarks[RIGHT_EYE_POINTS + LEFT_EYE_POINTS]

                xmax = 0
                ymax = 0
                xmin = 100000
                ymin = 100000

                # 포인트 출력
                for idx, point in enumerate(landmarks_display):
                    pos = (point[0, 0], point[0, 1])

                    if point[0, 0] > xmax:
                        xmax = point[0, 0]
                    if point[0, 0] < xmin:
                        xmin = point[0, 0]
                    if point[0, 1] > ymax:
                        ymax = point[0, 1]
                    if point[0, 1] < ymin:
                        ymin = point[0, 1]

                    eyeimg = frame[ymin - 30:ymax + 30, xmin - 30:xmax + 30]

                    try:

                        eyegray = cv2.cvtColor(eyeimg, cv2.COLOR_BGR2GRAY)
                        equ = cv2.equalizeHist(eyegray)
                        thres = cv2.inRange(equ, 0, 20)
                        kernel = np.ones((3, 3), np.uint8)

                        # print dlib points
                        cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

                        cv2.imshow("a", eyeimg)

                        # /------- removing small noise inside the white image ---------/#
                        dilation = cv2.dilate(thres, kernel, iterations=2)
                        # /------- decreasing the size of the white region -------------/#
                        erosion = cv2.erode(dilation, kernel, iterations=3)
                        # /-------- finding the contours -------------------------------/#
                        image, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        # --------- checking for 2 contours found or not ----------------#
                        if len(contours) == 2:
                            numerator += 1
                            # ------ finding the centroid of the contour ----------------#
                            M1 = cv2.moments(contours[1])

                            if M1['m00'] != 0:
                                cx = int(M1['m10'] / M1['m00'])
                                cy = int(M1['m01'] / M1['m00'])
                                cv2.circle(eyeimg, (cx, cy), 2, (0, 0, 255), thickness=-1)  # red point
                        # print cx,cy

                        # -------- checking for one countor presence --------------------#
                        elif len(contours) == 1:
                            numerator += 1
                            # img = cv2.drawContours(roi, contours, 0, (0,255,0), 3)

                            # ------- finding centroid of the countor ----#
                            M1 = cv2.moments(contours[0])
                            if M1['m00'] != 0:
                                cx = int(M1['m10'] / M1['m00'])
                                cy = int(M1['m01'] / M1['m00'])

                                # print cx,cy
                                cv2.circle(eyeimg, (cx, cy), 2, (0, 0, 255), thickness=-1)  # red point
                                print(str(cx) + "," + str(cy))
                        else:
                            denominator += 1
                        # print "iris not detected"
                        x1 = int(x + w / 1.7) + 1  # -- +1 is done to hide the green color
                        x2 = int(x + w / 1.3)
                        y1 = int(y + h / 3.3) + 1
                        y2 = int(y + h / 2.2)

                        eyeimg = frame[y1:y2, x1:x2]
                        eyegray = cv2.cvtColor(eyeimg, cv2.COLOR_BGR2GRAY)
                        equ = cv2.equalizeHist(eyegray)
                        thres = cv2.inRange(equ, 0, 20)
                        kernel = np.ones((3, 3), np.uint8)

                        # /------- removing small noise inside the white image ---------/#
                        dilation = cv2.dilate(thres, kernel, iterations=2)
                        # /------- decreasing the size of the white region -------------/#
                        erosion = cv2.erode(dilation, kernel, iterations=3)
                        # /-------- finding the contours -------------------------------/#
                        image, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        # --------- checking for 2 contours found or not ----------------#
                        if len(contours) == 2:
                            numerator += 1
                            # img = cv2.drawContours(roi, contours, 1, (0,255,0), 3)
                            # ------ finding the centroid of the contour ----------------#
                            M1 = cv2.moments(contours[1])
                            # print M['m00']
                            # print M['m10']
                            # print M['m01']

                            if M1['m00'] != 0:
                                cx = int(M1['m10'] / M1['m00'])
                                cy = int(M1['m01'] / M1['m00'])
                                cv2.line(eyeimg, (cx, cy), (cx, cy), (0, 0, 255), 3)  # red point
                        # print cx,cy
                        # -------- checking for one countor presence --------------------#
                        elif len(contours) == 1:
                            numerator += 1
                            # img = cv2.drawContours(roi, contours, 0, (0,255,0), 3)

                            # ------- finding centroid of the countor ----#
                            M1 = cv2.moments(contours[0])
                            if M1['m00'] != 0:
                                cx = int(M1['m10'] / M1['m00'])
                                cy = int(M1['m01'] / M1['m00'])

                                # print cx,cy
                                cv2.circle(eyeimg, (cx, cy), 2, (0, 0, 255), thickness=-1)  # red point
                                print(str(cx) + "," + str(cy))
                        else:
                            denominator += 1
                        # print "iris not detected"

                        ran = x2 - x1
                        mid = ran / 2
                        """
                        if cx < mid - 5:
                            print("looking left")
                        elif cx > mid + 5:
                            print("looking right")
                        elif mid - 5 < cx < mid + 5:
                            print("looking infront")"""
                    except cv2.error:
                        print("no img")


    return frame

# 웹캠에서 이미지 가져오기
video_capture = cv2.VideoCapture(0)

while True:
    # 웹캠 이미지를 프레임으로 자름
    _, frame = video_capture.read()
    # 그리고 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 만들어준 얼굴 눈 찾기
    canvas = detect(gray, frame)
    # 찾은 이미지 보여주기
    cv2.imshow("haha", canvas)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 끝
video_capture.release()
cv2.destroyAllWindows()
