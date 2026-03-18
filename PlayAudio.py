import os

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
import pygame

pygame.mixer.init()

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
#vid = cv2.VideoCapture(0)

camera_open = False

kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color|PyKinectV2.FrameSourceTypes_Depth
)

width, height = 1920, 1080

#vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


#playback = Playback()
#playback.stop()

previousPredictedGesture = ""
predictedGesture = None
predictedGestureCount = 0
confidentGesture = ""


root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
root.geometry('1920x1080')
filename = "Initial"

label_widget = Label(root)
label_widget.pack()

def select_file():
    possibleFiletypes = (
        ('Audio Files', '*.wav *.mp3'),
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=possibleFiletypes)

    if not filename:
        CurrentFile['text'] = "No Audiofile Selected"
        pause_button.config(state=tk.DISABLED)
        play_button.config(state=tk.DISABLED)
    
    else:
        pause_button.config(state=tk.NORMAL)
        play_button.config(state=tk.DISABLED)
        CurrentFile['text'] = os.path.basename(os.path.normpath(filename))
        #playback.load_file(filename)
        #playback.play()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

def pause_command():
    pygame.mixer.music.pause()

    pause_button.config(state=tk.DISABLED)
    play_button.config(state=tk.NORMAL)


def play_command():
    pygame.mixer.music.unpause()
    pause_button.config(state=tk.NORMAL)
    play_button.config(state=tk.DISABLED)

def on_camera_button_click():
    global camera_open
    camera_open = True
    threading.Thread(target=update_camera_loop).start()

def update_camera_loop():
    while camera_open == True:
        update_camera()

def update_camera():
    if(camera_open == False):
        webcamPanel.config(image=img)

    else:
        if kinect.has_new_color_frame():
            rawColourFrame = kinect.get_last_color_frame()
            shapedColourFrame = rawColourFrame.reshape((1080,1920,4))
            shapedColourFrame = shapedColourFrame[:,:,:3]
            shapedColourFrame = cv2.cvtColor(shapedColourFrame, cv2.COLOR_RGB2BGR)
            convertedColourFrame = Image.fromarray(shapedColourFrame)
            convertedColourFrame = ImageTk.PhotoImage(image=convertedColourFrame)
            
            webcamPanel.config(image=convertedColourFrame)
            webcamPanel.image = convertedColourFrame
    

def close_camera():
    global camera_open
    camera_open = False
    webcamPanel.config(image=img)

def on_gesture_button_click():
    global camera_open
    threading.Thread(target=gesture_recognition_loop).start()

def gesture_recognition_loop():
    while camera_open == True:
        gesture_recognition()

def gesture_recognition():
    global previousPredictedGesture, predictedGesture, predictedGestureCount, confidentGesture, camera_open
    #time.sleep(0.35)
    if (kinect.has_new_depth_frame() and camera_open == True):
        depth_frame = kinect.get_last_depth_frame()
        depth_frame = np.reshape(depth_frame, (424, 512))
        valid = depth_frame[depth_frame > 0]
        threshold = np.percentile(valid, 3) + 10


        depth_frame = np.array(depth_frame, dtype=np.float32)
        depth_frame[depth_frame > threshold] = 0
        print ("Min depth value: ", depth_frame[depth_frame > 0].min())
        nonZeroPixels = cv2.countNonZero(depth_frame)
        if((nonZeroPixels < 3000) or (depth_frame[depth_frame > 0].min() > 800)):
            print("No hand detected")
            label['text'] = "No hand detected"
                
        else:
            #plt.imshow(depth_frame, cmap='gray')
            #plt.clim(0, 700)
            #plt.colorbar()
            #plt.show()
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



            model_prediction = model.predict(hand_crop.reshape(1, 128, 128, 1))
            predictedGesture = gestures[model_prediction.argmax()]
            print(predictedGesture)
        
            if(predictedGesture == previousPredictedGesture):
                if(predictedGestureCount >= 2):
                    previousConfidentGesture = confidentGesture
                    confidentGesture = predictedGesture
                    label['text'] = confidentGesture

                    # For gestures that should only be triggered once, i.e playing, pausing
                    if(confidentGesture != previousConfidentGesture):
                        match predictedGesture:
                            case "Fist":
                                pygame.mixer.music.pause()
                                pass
                            case "Five Fingers":
                                pygame.mixer.music.unpause()
                                pass
                    
                    # For gestures that should be continuously triggered, i.e volume control, skipping through the track
                    else:
                        curVolume = pygame.mixer.music.get_volume()
                        match predictedGesture:
                            
                            case "Single Finger":
                                
                                pygame.mixer.music.set_volume(curVolume + 0.2)
                                pass
                            case "Two Fingers":
                                pygame.mixer.music.set_volume(curVolume - 0.2)
                                pass
                        print("Volume: ", curVolume)
                        curVolume = pygame.mixer.music.get_volume()
                        CurrentVolume['text'] = "Volume: " + str(curVolume)    
                else:
                    predictedGestureCount += 1
            else:
                predictedGestureCount = 0
                previousPredictedGesture = predictedGesture
        
    else:
        print("No depth frame")
    
    
# Sidebar
sidebar = tk.Frame(root, width=200, bg='lightgray')

info = tk.Frame(sidebar, bg='lightblue', height=100)

CurrentFile = tk.Label(info, text="No Audiofile Selected")
CurrentFile.pack()

getInitVolume = pygame.mixer.music.get_volume()
CurrentVolume = tk.Label(info, text="Volume: " + str(getInitVolume))
CurrentVolume.pack()

info.pack(fill=tk.X)

open_button = Button(
    sidebar,
    text='Open a File',
    command=select_file
)
open_button.pack()


pause_button = Button(
    sidebar,
    text='Pause',
    command=pause_command,
    state=tk.DISABLED
)
pause_button.pack()

play_button = Button(
    sidebar,
    text='Play',
    command=play_command,
    state=tk.DISABLED
)
play_button.pack()

feed_button = Button(sidebar, 
                     text="Open Camera", 
                     command=on_camera_button_click)
feed_button.pack()

close_camera_button = Button(sidebar,
                      text="Close Camera",
                      command = close_camera)
close_camera_button.pack()

gesture_recognition_button = Button(sidebar,
                        text="Start Gesture Recognition",
                        command = on_gesture_button_click)
gesture_recognition_button.pack()

sidebar.pack(side=tk.LEFT, fill=tk.Y)




#canvas.pack()

print(filename)

img = Image.open("CameraDisabled.png")
img = img.resize((width, height))
img = ImageTk.PhotoImage(image=img)

webcamPanel = Label(root, image=img)
webcamPanel.pack()
root.mainloop()



        