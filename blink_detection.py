# import the necessary packages
from scipy.spatial import distance
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
from datetime import datetime
import dlib
import cv2

class Blink:
    def __init__(self, startTime, duration):
        self.startTime = startTime
        self.duration = duration
    
STARTING_TIME = datetime.now()
STATE = "RUNNING"
TOTAL_BLINKS = 0
EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 2
COUNTER = 0
ASPECT_RATIO_VECTOR = []

def getPauseDuration():
    duration = getDuration(PAUSE_START_TIME, datetime.now())
    if (duration > 120):
        return True
    EXIT_TIME = STARTING_TIME + duration
    return False


def getDuration(date1, date2):
    timeDelta = date2 - date1
    totalSeconds = timeDelta.total_seconds()
    return totalSeconds


def eyeAspectRatio(eye):
    # Representation of Human eye
    '''
        p2    p3
    p1            p4 
        p6    p5

    '''
    # Each eye is represented by an array of above 6 points (p1 -> p6) in clockwise direction
    # Eye Aspect Ratio = (||p2 - p6|| + ||p3 - p5|| )/ 2 * ||p4 - p1||

    # Euclidean distances between vertical eye landmarks
    d1 = distance.euclidean(eye[1], eye[5])
    d2 = distance.euclidean(eye[2], eye[4])

    # Euclidean distance between horizontal eye landmarks
    d3 = distance.euclidean(eye[0], eye[3])

    eye_aspect_ratio = (d1 + d2) / (2.0 * d3)
    return eye_aspect_ratio


# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("INFO: loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# get indexes of the facial landmarks for the left and right eye
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("INFO: starting video stream thread...")
vs = VideoStream(0).start()
fileStream = False
time.sleep(1.0)

while True:
    # Read the frame, resize it and convert to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width = 450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = detector(gray, 0)

    if (len(faces) != 1):
        STATE = "PAUSED"
        print("WARNING: face count is not equal to 1. pausing frames... ")
        PAUSE_START_TIME = datetime.now()

    else:
        #if (STATE == "PAUSED" and getPauseDuration()):
            #break
        STATE = "RUNNING"
        face = faces[0]
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # extracting the left and right eye from the facial landmarks
        leftEye = shape[leftEyeStart: leftEyeEnd]
        rightEye = shape[rightEyeStart: rightEyeEnd]

        # calculating aspect ratios for both eyes
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        ASPECT_RATIO_VECTOR.append(ear)
        
        if (len(ASPECT_RATIO_VECTOR) >= 60):
            ASPECT_RATIO_VECTOR.pop(0)

        EYE_AR_THRESH = sum(ASPECT_RATIO_VECTOR)/len(ASPECT_RATIO_VECTOR) - 0.06
        
        
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL_BLINKS += 1
            COUNTER = 0



        cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINKS), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
