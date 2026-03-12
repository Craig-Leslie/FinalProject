import numpy as np
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

if (kinect.has_new_depth_frame):
    depth_frame = kinect.get_last_depth_frame