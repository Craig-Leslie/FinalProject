import keyboard
from just_playback import Playback
import time
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
from PIL import Image, ImageTk
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2
import threading

import keras
import numpy as np
import matplotlib.pyplot as plt

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

depth_frame = None
model = keras.models.load_model("my_model.h5")


#vid = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
vid = cv2.VideoCapture(0)

camera_open = False

kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color|PyKinectV2.FrameSourceTypes_Depth
)

width, height = 1920, 1080

vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


playback = Playback()
playback.stop()



root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
root.geometry('300x150')
filename = "Initial"

label_widget = Label(root)
label_widget.pack()

def select_file():
    filetypes = (
        ('Audio Files', '*.wav *.mp3'),
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    if not filename:
        label['text'] = "No file selected"
        pause_button.config(state=tk.DISABLED)
        play_button.config(state=tk.DISABLED)
    
    else:
        pause_button.config(state=tk.NORMAL)
        play_button.config(state=tk.DISABLED)
        label['text'] = filename
        playback.load_file(filename)
        playback.play()

def pause_command():
    playback.pause()
    pause_button.config(state=tk.DISABLED)
    play_button.config(state=tk.NORMAL)


def play_command():
    playback.play()
    pause_button.config(state=tk.NORMAL)
    play_button.config(state=tk.DISABLED)


def open_camera():
    """"
    _, frame = vid.read()

    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)

    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(10, open_camera)

    """
    
    
        #time.sleep(0.5)
    if kinect.has_new_color_frame():

        #print("New frame")
        rawColourFrame = kinect.get_last_color_frame()
        #
        shapedColourFrame = rawColourFrame.reshape((1080,1920,4))
        shapedColourFrame = shapedColourFrame[:,:,:3]
        shapedColourFrame = cv2.cvtColor(shapedColourFrame, cv2.COLOR_BGR)
        convertedColourFrame = Image.fromarray(shapedColourFrame)
        convertedColourFrame = ImageTk.PhotoImage(image=convertedColourFrame)
        
        webcamPanel.config(image=convertedColourFrame)
        webcamPanel.image = convertedColourFrame


        #cv2.imshow("Kinect RGB", frame)
    
    label_widget.after(10, open_camera)

def close_camera():
    global camera_open
    camera_open = False

def gesture_recognition():
    time.sleep(0.5)
    if (kinect.has_new_depth_frame()):
        depth_frame = kinect.get_last_depth_frame()
    
    else:
        return
    
    depth_frame = np.reshape(depth_frame, (424, 512))
    valid = depth_frame[depth_frame > 0]
    threshold = np.percentile(valid, 3) + 10


    depth_frame = np.array(depth_frame, dtype=np.float32)
    depth_frame[depth_frame > 1000] = 0
    depth_frame[depth_frame > threshold] = 0

    nonZeroPixels = cv2.countNonZero(depth_frame)
    if(nonZeroPixels < 3000):
        print("No hand detected")
        label['text'] = "No hand detected"
        return       
    
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

    #plt.imshow(hand_crop, cmap='gray')
    #plt.clim(0, 700)
    #plt.colorbar()
    #plt.show()

    model_prediction = model.predict(hand_crop.reshape(1, 128, 128, 1))
    print(model_prediction)
    print(gestures[model_prediction.argmax()])
    label['text'] = gestures[model_prediction.argmax()]
    label_widget.after(50, gesture_recognition)

open_button = Button(
    root,
    text='Open a File',
    command=select_file
)


pause_button = Button(
    root,
    text='Pause',
    command=pause_command,
    state=tk.DISABLED
)

play_button = Button(
    root,
    text='Play',
    command=play_command,
    state=tk.DISABLED
)

feed_button = Button(root, 
                     text="Open Camera", 
                     command=open_camera)

close_camera_button = Button(root,
                      text="Close Camera",
                      command = close_camera)

gesture_recognition_button = Button(root,
                        text="Start Gesture Recognition",
                        command = gesture_recognition)
gesture_recognition_button.pack(expand=True)

canvas = tk.Canvas(root)

feed_button.pack()

open_button.pack(expand=True)
close_camera_button.pack(expand=True)

pause_button.pack()
play_button.pack()
canvas.pack()
label = tk.Label(root, text="Audio Path")
label.pack()
print(filename)

img = Image.open("image1.png")
img = img.resize((width, height))
img = ImageTk.PhotoImage(image=img)

webcamPanel = Label(root, image=img)
webcamPanel.pack()
root.mainloop()



        