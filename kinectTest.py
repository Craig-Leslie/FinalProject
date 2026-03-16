import time

import cv2
import cv2
import numpy as np
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2
import matplotlib.pyplot as plt
import keras

gestures = {
    0: "Fist",
    1: "Single Finger",
    2: "Two Fingers",
    3: "Three Fingers",
    4: "Four Fingers",
    5: "Five Fingers",
    6: "Surfer",
    7: "L",
    8: "Thumbs Up",
    9: "Rock On"
}

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
depth_frame = None

model = keras.models.load_model("my_model.h5")

if(not kinect):
    print("Kinect not found")

else:
    print("Kinect found")

while (depth_frame is None):

    if (kinect.has_new_depth_frame()):
        depth_frame = kinect.get_last_depth_frame()

    else:
        time.sleep(0.1)

    #depth_frame = depth_frame.astype(np.uint16)

print(depth_frame)
depth_frame = np.reshape(depth_frame, (424, 512))


print(depth_frame.shape)
plt.imshow(depth_frame, cmap='gray')
plt.colorbar()
plt.show()#depthArray = depthArray.reshape((480, 640))

k = 0
#threshold = (np.min(depth_frame[depth_frame>k])) + 75
#print(threshold)
valid = depth_frame[depth_frame > 0]
threshold = np.percentile(valid, 3) + 10


depth_frame = np.array(depth_frame, dtype=np.float32)
#depth[depth == 0] = np.nan
depth_frame[depth_frame > 1000] = 0
depth_frame[depth_frame > threshold] = 0

nonZeroPixels = cv2.countNonZero(depth_frame)
if(nonZeroPixels < 3000):
    print("No hand detected")
    exit()

plt.imshow(depth_frame, cmap='gray')
plt.colorbar()
plt.show()

mask = depth_frame > 0
mask = mask.astype(np.uint8)


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

clean_mask = (labels == largest_label).astype(np.uint8)

depth_frame = depth_frame * clean_mask
ys, xs = np.where(depth_frame>0)


padding = 20

ymin, ymax = max(0, ys.min() - padding), max(0, ys.max() + padding)
xmin, xmax = max(0, xs.min() - padding), max(0, xs.max() + padding)

print(ymin, ymax, xmin, xmax)
hand_crop = depth_frame[ymin:ymax, xmin:xmax]
hand_crop = cv2.resize(hand_crop, (128, 128))

print(hand_crop.max())
plt.imshow(hand_crop, cmap='gray')
plt.clim(0, 700)
plt.colorbar()
plt.show()




model_prediction = model.predict(hand_crop.reshape(1, 128, 128, 1))
print(model_prediction)
print(gestures[model_prediction.argmax()])