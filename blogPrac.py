import numpy as np
import cv2

# 찾고자하는 것의 cascade classifier 를 등록
# 경로는 상대경로로 바뀔 수 있음
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

""" 
    def = haar를 이용 얼굴과 눈을 찾는 함수
    input = 그레이 스케일 이미지
    output = 얼굴과 눈에 사각형이 그려진 이미지 프레임
"""
def detect(gray,frame):
    numerator = 0
    denominator = 0

    # 등록한 Cascade classifier 를 이용 얼굴을 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100),flags=cv2.CASCADE_SCALE_IMAGE)

    # 얼굴에 사각형을 그리고 눈을 찾자
    for (x, y, w, h) in faces:
        # 얼굴: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 255 0 0 , 굵기 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 이미지를 얼굴 크기 만큼 잘라서 그레이스케일 이미지와 컬러이미지를 만듬
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # 등록한 Cascade classifier 를 이용 눈을 찾음(얼굴 영역에서만)
        eyes = eyeCascade.detectMultiScale(face_gray, 1.1, 3)

        # left eye #
        # ---------- horizontal lower line ----------------#
        cv2.line(frame, (int(x + w / 4.2), int(y + h / 2.2)), (int(x + w / 2.5), int(y + h / 2.2)), (0, 255, 0), 1)
        # ----------- horizontal upper line ------#
        cv2.line(frame, (int(x + w / 4.2), int(y + h / 3.3)), (int(x + w / 2.5), int(y + h / 3.3)), (0, 255, 0), 1)
        # ---------- vertical left line ----------#
        cv2.line(frame, (int(x + w / 4.2), int(y + h / 3.3)), (int(x + w / 4.2), int(y + h / 2.2)), (0, 255, 0), 1)
        # ---------- vertical right line---------------#
        cv2.line(frame, (int(x + w / 2.5), int(y + h / 3.3)), (int(x + w / 2.5), int(y + h / 2.2)), (0, 255, 0), 1)

        # right eye #
        # ---------- horizontal lower line ----------------#
        cv2.line(frame, (int(x + w / 1.3), int(y + h / 2.2)), (int(x + w / 1.7), int(y + h / 2.2)), (0, 255, 0), 1)
        # ----------- horizontal upper line ------#
        cv2.line(frame, (int(x + w / 1.3), int(y + h / 3.3)), (int(x + w / 1.7), int(y + h / 3.3)), (0, 255, 0), 1)
        # ---------- vertical left line ----------#
        cv2.line(frame, (int(x + w / 1.3), int(y + h / 3.3)), (int(x + w / 1.3), int(y + h / 2.2)), (0, 255, 0), 1)
        # ---------- vertical right line---------------#
        cv2.line(frame, (int(x + w / 1.7), int(y + h / 3.3)), (int(x + w / 1.7), int(y + h / 2.2)), (0, 255, 0), 1)

        # ------------ estimation of distance of the human from camera--------------#
        d = 10920.0 / float(w)





        # -------- coordinates of interest --------------#
        x1 = int(x + w / 4.2) + 1  # -- +1 is done to hide the green color
        x2 = int(x + w / 2.5)
        y1 = int(y + h / 3.3) + 1
        y2 = int(y + h / 2.2)

        roi1 = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
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

            # ------ finding the centroid of the contour ----------------#
            M1 = cv2.moments(contours[1])

            if M1['m00'] != 0:
                cx = int(M1['m10'] / M1['m00'])
                cy = int(M1['m01'] / M1['m00'])
                cv2.line(roi1, (cx, cy), (cx, cy), (0, 0, 255), 3)  # red point
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
                cv2.line(roi1, (cx, cy), (cx, cy), (0, 0, 255), 3)  # red point
                print(str(cx) + "," + str(cy))
        else:
            denominator += 1
        # print "iris not detected"
        x1 = int(x + w / 1.7) + 1  # -- +1 is done to hide the green color
        x2 = int(x + w / 1.3)
        y1 = int(y + h / 3.3) + 1
        y2 = int(y + h / 2.2)

        roi1 = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
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
                cv2.line(roi1, (cx, cy), (cx, cy), (0, 0, 255), 3)  # red point
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
                cv2.line(roi1, (cx, cy), (cx, cy), (0, 0, 255), 3)  # red point
                print(str(cx) + "," + str(cy))
        else:
            denominator += 1
        # print "iris not detected"

        ran = x2 - x1
        mid = ran / 2
        if cx < mid - 5:
            print("looking left")
        elif cx > mid + 5:
            print("looking right")
        elif mid - 5 < cx < mid + 5:
            print("looking infront")

        # 눈: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 0 255 0 , 굵기 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

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