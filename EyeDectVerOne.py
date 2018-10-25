import numpy as np
import cv2
import dlib
import math
import time
import sys

FACECASCADE_PATH = "haarcascade_frontalface_default.xml"
CUSTOMCASCADE_PATH = "cascade2000_30.xml"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

faceCascade = cv2.CascadeClassifier(FACECASCADE_PATH)
customCascade = cv2.CascadeClassifier(CUSTOMCASCADE_PATH)
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# 얼굴의 각 구역의 포인트들을 구분해 놓기
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

detectrunning = 0
count = 0
userValidPos = ()
userlefteye = ()
userrighteye = ()
userLeftLand = ()
userRightLand = ()
posStayCount = 0
prevMousePos = ()
mouseFocus = 0

def checkMouseFocus(pos):
    global posStayCount
    global prevMousePos
    global mouseFocus

    countThr = 5
    moveThr = 20
    if pos != (0,0):
        if prevMousePos == ():
            prevMousePos = pos
            return mouseFocus
        elif posStayCount < countThr:
            posStayCount = posStayCount + 1
            if pos[0] - moveThr < prevMousePos[0] < pos[0] + moveThr and pos[1] - moveThr < prevMousePos[1] < pos[1] + moveThr:
               mouseFocus = 1
            else:
                print("not")
                posStayCount = 0
                mouseFocus = 0
            prevMousePos = pos
            return 0
        elif posStayCount == countThr:
            print("cl")
            posStayCount = 0
            prevMousePos = ()
            return mouseFocus


def detectEye(landmarks_display, dir):

    xmin = sys.maxsize
    xmax = 0
    ymin = sys.maxsize
    ymax = 0

    global userlefteye
    global userrighteye
    for idx, point in enumerate(landmarks_display):
        pos = (point[0, 0], point[0, 1])
        if point[0, 0] > xmax:
            xmax = point[0, 0] + 20
        if point[0, 0] < xmin:
            xmin = point[0, 0] - 20
        if point[0, 1] > ymax:
            ymax = point[0, 1] + 20
        if point[0, 1] < ymin:
            ymin = point[0, 1] - 20
    #left
    if dir == 0:
        if userlefteye == ():
            userlefteye = (xmin,xmax,ymin,ymax)

    if dir == 1:
        if userrighteye == ():
            userrighteye = (xmin,xmax,ymin,ymax)




def pupilDetect(frame,landmarks_display,eyepos):

    global userlefteye
    global userrighteye

    if userlefteye == () or userrighteye == ():
        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            if point[0, 0] > xmax:
                xmax = point[0, 0] + 20
            if point[0, 0] < xmin:
                xmin = point[0, 0] - 20
            if point[0, 1] > ymax:
                ymax = point[0, 1] + 20
            if point[0, 1] < ymin:
                ymin = point[0, 1] - 20

        # left
        if dir == 0:
            userlefteye = [xmin, xmax, ymin, ymax]
        # right
        elif dir == 1:
            userrighteye = [xmin, xmax, ymin, ymax]



def pupilDetect(frame, landmarks_display, eyepos):
    numerator = 0
    denominator = 0
    ret = (0, 0)
    try:
        eyeimg = frame[eyepos[2]:eyepos[3], eyepos[0]:eyepos[1]]
        eyeimg = cv2.resize(eyeimg, None, fx=15, fy=15, interpolation=cv2.INTER_CUBIC)
        eyegray = cv2.cvtColor(eyeimg, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(eyegray)
        thres = cv2.inRange(equ, 0, 20)
        kernel = np.ones((3, 3), np.uint8)

        # /------- decreasing the size of the white region -------------/#
        erosion = cv2.erode(thres, kernel, iterations=20)
        # /------- removing small noise inside the white image ---------/#
        dilation = cv2.dilate(erosion, kernel, iterations=45)
        # /------- decreasing the size of the white region -------------/#
        erosion = cv2.erode(dilation, kernel, iterations=30)

        # /-------- finding the contours -------------------------------/#
        image, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 1:
            numerator += 1
            # img = cv2.drawContours(roi, contours, 0, (0,255,0), 3)

            # ------- finding centroid of the countor ----#
            M1 = cv2.moments(contours[0])
            if M1['m00'] != 0:
                cx = int(M1['m10'] / M1['m00'])
                cy = int(M1['m01'] / M1['m00'])

                # print cx,cy
                cv2.circle(eyeimg, (cx, cy), 2, (0, 0, 255), thickness=-1)  # red point
                ret = (cx, cy)
                #pyautogui.moveTo(cx, cy)
        else:
            denominator += 1

        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

        #cv2.imshow("b", eyeimg)
        #cv2.imshow("a", thres)

    except cv2.error:
        print("no img")

    return ret

def faceDetect(faces, gray, frame):
    global userValidPos

    # 얼굴에서 랜드마크를 찾자
    for (x, y, w, h) in faces:
        # 얼굴: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 255 0 0 , 굵기 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # face를 이미지로 잘라서 red star 인식하기
        faceImg = frame[y:y + h, x:x + w]
        faceGray = gray[y:y + h, x:x + w]
        # find red star
        customPoint = customCascade.detectMultiScale(faceGray)

        if customPoint != ():
            # if found valid face save the pos
            userValidPos = (x, y, w, h)
            print("user pos: ", userValidPos)

            for (px, py, pw, ph) in customPoint:
                cv2.rectangle(faceImg, (px, py), (px + pw, py + ph), (0, 255, 0), 2)

""" 
    def = dlib를 이용 얼굴과 눈을 찾는 함수
    input = 그레이 스케일 이미지
    output = 얼굴 중요 68개의 포인트 에 그려진 점 + 이미지
"""

def detect(gray, frame):
    global userValidPos
    global mouseFocus
    global userlefteye
    global userrighteye
    global userLeftLand
    global userRightLand

    # init the face pos
    if userValidPos == ():
        # 일단, 등록한 Cascade classifier 를 이용 얼굴을 찾음
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        faceDetect(faces, gray, frame)
    else:
        # 얼굴: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 255 0 0 , 굵기 2)
        cv2.rectangle(frame, (userValidPos[0], userValidPos[1]), (userValidPos[0] + userValidPos[2], userValidPos[1] + userValidPos[3]), (255, 0, 0), 2)
        if userLeftLand == () and userRightLand == ():
            # set the eye pos
            # 오픈 CV 이미지를 dlib용 사각형으로 변환하고
            dlib_rect = dlib.rectangle(int(userValidPos[0]), int(userValidPos[1]), int(userValidPos[0] + userValidPos[2],), int(userValidPos[1] + userValidPos[3]))
            # 랜드마크 포인트들 지정
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
            userLeftLand = landmarks[LEFT_EYE_POINTS]
            userRightLand = landmarks[RIGHT_EYE_POINTS]
        else:
            if userLeftLand != ():
                detectEye(userLeftLand, 0)
            if userRightLand != ():
                detectEye(userRightLand, 1)

        # detect pupil
        if userlefteye != ():
            cv2.rectangle(frame,(userlefteye[0],userlefteye[2]),(userlefteye[1],userlefteye[3]),(0,255,0),2)
            pupilDetect(frame,userLeftLand,userlefteye)
        if userrighteye != ():
            cv2.rectangle(frame, (userrighteye[0], userrighteye[2]), (userrighteye[1], userrighteye[3]), (0, 255, 0), 2)
            pupilDetect(frame,userRightLand,userrighteye)
    return frame

# 웹캠에서 이미지 가져오기
video_capture = cv2.VideoCapture(0)
framecount = 0

while True:
    # 웹캠 이미지를 프레임으로 자름
    _, frame = video_capture.read()
    # 좌우반전
    frame = cv2.flip(frame, 1)

    # 그리고 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 만들어준 얼굴 눈 찾기
    canvas = detect(gray, frame)
    # 찾은 이미지 보여주기
    cv2.imshow("haha", canvas)

    kb = cv2.waitKey(1) & 0xFF

    if kb == ord('r'):
        print("reset")
        userValidPos = ()
        userlefteye = ()
        userrighteye = ()
        userLeftLand = ()
        userRightLand =()
        posStayCount = 0
        prevMousePos = ()
        mouseFocus = 0

    # q를 누르면 종료
    if kb == ord('q'):
        break

# 끝
video_capture.release()
cv2.destroyAllWindows()