import cv2
import numpy as np
import argparse

import math


def detect(gray,frame):
    """ input = greyscale imeage
        output = boxes
    """
    # faces are the tuples of 4 numbers
    # x,y => upperleft corner coordinates of face
    # width(w) of rectangle in the face
    # height(h) of rectangle in the face
    # grey means the input image to the detector
    # 1.3 is the kernel size or size of image reduced when applying the detection
    # 5 is the number of neighbors after which we accept that is a face

    #faces = faceCascade.detectMultiScale(gray, 1.3,5)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Now iterate over the faces and detect eyes


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Arguements => image, top-left coordinates, bottomright coordinates, color, rectangle border thickness

        # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # Detect eyes now
        eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 3)
        # Detect pupil
        # Now draw rectangle over the eyes
        for (ex, ey, ew, eh) in eyes:
            pup = roi_color[ey:ey + eh, ex:ex + ew]

            """
            pup = cv2.medianBlur(pup,5)
            try:
                cpup = cv2.cvtColor(pup,cv2.COLOR_GRAY2BGR)
                circles = cv2.HoughCircles(pup, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=25, minRadius=10, maxRadius=30)
                print("success")
                circles = np.uint16(np.around(circles))
                drawpupil(circles,cpup)
            except cv2.error:
                print("no file")
            """
            #cv2.imshow("a",pup)
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])

        landmarks_display = landmarks[RIGHT_EYE_POINTS]


        xmax = 0
        ymax = 0
        xmin = 100000
        ymin = 100000
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

            eyeimg = frame[ymin-30:ymax+30, xmin-30:xmax+30]
            try:
                eyeimg = cv2.GaussianBlur(eyeimg,(3,3),0)
                eyegray = cv2.cvtColor(eyeimg,cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(eyegray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=25, minRadius=0, maxRadius=0)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        print((i[0], i[1]), i[2])
                        #cv2.circle(eyeimg, (i[0], i[1]), i[2], color=(255, 255, 0), thickness=1)
                        cv2.circle(eyeimg, (i[0], i[1]), 1, color=(0, 0, 255), thickness=3)
                cv2.imshow("a",eyeimg)
            except cv2.error:
                print("no img")

            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

    return frame

# Capture video from webcam
video_capture = cv2.VideoCapture(0)
# Run the infinite loop
while True:
    # Read each frame
    _, frame = video_capture.read()
    # Convert frame to grey because cascading only works with greyscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Call the detect function with grey image and colored frame
    canvas = detect(gray, frame)
    # Show the image in the screen
    cv2.imshow("Video", canvas)
    # Put the condition which triggers the end of program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()