import numpy as np
import cv2
from pykinect2 import PyKinectRuntime, PyKinectV2
import time

kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color
)

while True:
    if kinect.has_new_color_frame():
        frame = kinect.get_last_color_frame()

        frame = frame.reshape((1080,1920,4))
        frame = frame[:,:,:3]

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow("Kinect RGB", frame)

    if cv2.waitKey(1) == 27:
        print("Exiting...")
        break