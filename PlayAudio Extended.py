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

from PIL import Image, ImageTk
import sounddevice as sd

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
    9: "Rock On",
    10: None
}

depth_frame = None
model = keras.models.load_model("my_model.h5")


#vid = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
#vid = cv2.VideoCapture(0)

camera_open = False
depth_open = False

kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color|PyKinectV2.FrameSourceTypes_Depth
)

width, height = 1920, 1080

#vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


#playback = Playback()
#playback.stop()


audioMode = "None"
previousPredictedGesture = ""
predictedGesture = None
predictedGestureCount = 0
confidentGesture = ""

left_channels = []
left_stems = []
right_channels = []

left_frame_previous_predicted_gesture = None
right_frame_previous_predicted_gesture = None

left_frame_predicted_gesture = None
right_frame_predicted_gesture = None

left_frame_predicted_gesture_count = 0
right_frame_predicted_gesture_count = 0

left_frame_confident_gesture = None
right_frame_confident_gesture = None

previous_left_frame_confident_gesture = None
previous_right_frame_confident_gesture = None


root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
root.geometry('1920x1080')
filename = "Initial"

label_widget = Label(root)
label_widget.pack()

def select_file():
    
    global audioMode, channels
    possibleFiletypes = (
        ('Audio Files', '*.wav *.mp3'),
    )

    filenames = fd.askopenfilenames(
        title='Open a file',
        initialdir='/',
        filetypes=possibleFiletypes)
    

    if len(filenames) == 0:
        CurrentFile['text'] = "No Audiofile Selected"
        pause_button.config(state=tk.DISABLED)
        play_button.config(state=tk.DISABLED)
    
    elif len(filenames) == 1:
        audioMode = "Single"
        music_file = filenames[0]
        #print(len(filenames))
        pause_button.config(state=tk.NORMAL)
        play_button.config(state=tk.DISABLED)
        #print(filenames[0])
        CurrentFile['text'] = os.path.basename(os.path.normpath(music_file))
        #playback.load_file(filename)
        #playback.play()
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
    
    elif 8 > len(filenames) > 1:
        audioMode = "Multi"
        for i, file in enumerate(filenames):
            stemText = tk.Label(gestureIcons, text=file.split("/")[-1])
            stemText.grid(row=2, column=i)
        pygame.mixer.set_num_channels(len(filenames))
        stems = [pygame.mixer.Sound(file) for file in filenames]
        CurrentFile['text'] = "Multiple Stems Loaded"
        channels = [pygame.mixer.Channel(i) for i in range(len(stems))]

        for i, channel in enumerate(channels):
            channel.play(stems[i])
            

    else:
        CurrentFile['text'] = "Too many files selected, please select 1-7 audio files"

    currentMode['text'] = "Current Mode: " + audioMode
            

def select_file_left():
    global audioMode, left_channels, left_stems
    possibleFiletypes = (
        ('Audio Files', '*.wav *.mp3'),
    )

    filenames = fd.askopenfilenames(
        title='Open a file',
        initialdir='/',
        filetypes=possibleFiletypes)
    

    if len(filenames) == 0:
        CurrentFile['text'] = "No Audiofile Selected"
        pause_button.config(state=tk.DISABLED)
        play_button.config(state=tk.DISABLED)
    
    elif len(filenames) == 1:
        CurrentFile['text'] = "Need at least 2 files for multi-stem mode"
    
    elif 8 > len(filenames) > 1:
        audioMode = "Multi"
        for i, file in enumerate(filenames):
            stemText = tk.Label(gestureIcons, text=file.split("/")[-1])
            stemText.grid(row=2, column=i)
        pygame.mixer.set_num_channels(len(filenames))
        left_stems = [pygame.mixer.Sound(file) for file in filenames]
        CurrentFile['text'] = "Multiple Stems Loaded"
        left_channels = [pygame.mixer.Channel(i) for i in range(len(left_stems))]
        print(left_channels)
        

            

    else:
        CurrentFile['text'] = "Too many files selected, please select 1-7 audio files"

    currentMode['text'] = "Current Mode: " + audioMode


def select_file_right():
    global audioMode, right_channels, left_stems
    possibleFiletypes = (
        ('Audio Files', '*.wav *.mp3'),
    )

    filenames = fd.askopenfilenames(
        title='Open a file',
        initialdir='/',
        filetypes=possibleFiletypes)
    

    if len(filenames) == 0:
        CurrentFile['text'] = "No Audiofile Selected"
        pause_button.config(state=tk.DISABLED)
        play_button.config(state=tk.DISABLED)
    
    elif len(filenames) == 1:
        CurrentFile['text'] = "Need at least 2 files for multi-stem mode"
    
    elif 8 > len(filenames) > 1:
        audioMode = "Multi"
        for i, file in enumerate(filenames):
            stemText = tk.Label(gestureIcons, text=file.split("/")[-1])
            stemText.grid(row=2, column=i)
        curChannels = pygame.mixer.get_num_channels()
        pygame.mixer.set_num_channels(curChannels + len(filenames))
        stems = [pygame.mixer.Sound(file) for file in filenames]
        CurrentFile['text'] = "Multiple Stems Loaded"
        right_channels = [pygame.mixer.Channel(i+len(left_channels)) for i in range(len(stems))]

        print(right_channels)
        
        both_channels = left_channels + right_channels
        both_stems = left_stems + stems
        for i, channel in enumerate(both_channels):
                channel.play(both_stems[i])
            

    else:
        CurrentFile['text'] = "Too many files selected, please select 1-7 audio files"

    currentMode['text'] = "Current Mode: " + audioMode
    
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
    global camera_open, depth_open
    camera_open = False
    depth_open = False
    webcamPanel.config(image=img)

def on_gesture_button_click():
    
    global camera_open, depth_open
    if(depth_open == False):
        depth_open = True
        threading.Thread(target=gesture_recognition_loop).start()

def gesture_recognition_loop():
    print("Starting Gesture Recognition Loop")
    while camera_open == True:
        gesture_recognition()

def CNN_Model(depth_frame):
        valid = depth_frame[depth_frame > 0]
        threshold = np.percentile(valid, 3) + 10
        depth_frame[depth_frame > threshold] = 0
        #print ("Min depth value: ", depth_frame[depth_frame > 0].min())
        nonZeroPixels = cv2.countNonZero(depth_frame)
        if((nonZeroPixels < 3000) or (depth_frame[depth_frame > 0].min() > 800)):
            #print("No hand detected")
            predictedGesture = None
            CurrentGesture['text'] = "No hand detected"
            return None, None
                
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

            #print(ymin, ymax, xmin, xmax)
            hand_crop = depth_frame[ymin:ymax, xmin:xmax]
            hand_crop = cv2.resize(hand_crop, (128, 128))


            model_prediction = model.predict(hand_crop.reshape(1, 128, 128, 1), verbose=0)
            predictedGestureValue = model_prediction.argmax()
            predictedGesture = gestures[model_prediction.argmax()]
            return predictedGestureValue, predictedGesture
        
def audio_operations(predictedGesture, previousPredictedGesture, confidentGesture, previousConfidentGesture, predictedGestureValue, channels, predictedGestureCount):
    print("Check 1")
    if(predictedGesture == previousPredictedGesture):
        print("Check 2")
        if(predictedGestureCount >= 4):
            print("Check 3")
            previousConfidentGesture = confidentGesture
            confidentGesture = predictedGesture

            print(confidentGesture)
            if(predictedGesture!=None):
                for i in range(len(imageText)):
                    if(i != predictedGestureValue):
                        imageText[i].config(bg="white")
                    elif(i == predictedGestureValue):
                        imageText[i].config(bg="pale green")

            CurrentGesture['text'] = confidentGesture

            if(confidentGesture != previousConfidentGesture):
                curGestureIndex = list(gestures.values()).index(confidentGesture)
                if((confidentGesture == "Rock On") | (predictedGesture == None)):
                    for channel in channels:
                        channel.set_volume(1)
                else:
                    for i, channel in enumerate(channels):
                        if(i != curGestureIndex):
                            channel.set_volume(0)
                        else:
                            channel.set_volume(1)                
        else:
            predictedGestureCount += 1
    else:
        predictedGestureCount = 0
        previousPredictedGesture = predictedGesture
        

    return previousPredictedGesture, confidentGesture, previousConfidentGesture, predictedGestureCount

def gesture_recognition():
    global previousPredictedGesture, predictedGesture, predictedGestureCount, confidentGesture, camera_open, left_frame_previous_predicted_gesture, right_frame_previous_predicted_gesture, left_frame_confident_gesture, right_frame_confident_gesture, previous_left_frame_confident_gesture, previous_right_frame_confident_gesture, left_frame_predicted_gesture_count, right_frame_predicted_gesture_count
    global left_channels, right_channels
    #time.sleep(0.35)
    if (kinect.has_new_depth_frame() and camera_open == True):
        depth_frame = kinect.get_last_depth_frame()
        depth_frame = np.reshape(depth_frame, (424, 512))
        


        depth_frame = np.array(depth_frame, dtype=np.float32)
        left_depth_frame = depth_frame[:,:depth_frame.shape[1]//2]
        right_depth_frame = depth_frame[:,depth_frame.shape[1]//2:]

        
        left_frame_prediction_values, left_frame_prediction = CNN_Model(left_depth_frame)
        right_frame_prediction_values, right_frame_prediction = CNN_Model(right_depth_frame)
        
        left_frame_previous_predicted_gesture, left_frame_confident_gesture, previous_left_frame_confident_gesture, left_frame_predicted_gesture_count = audio_operations(left_frame_prediction, left_frame_previous_predicted_gesture, left_frame_confident_gesture, previous_left_frame_confident_gesture, left_frame_prediction_values, left_channels, left_frame_predicted_gesture_count)
        right_frame_previous_predicted_gesture, right_frame_confident_gesture, previous_right_frame_confident_gesture, right_frame_predicted_gesture_count = audio_operations(right_frame_prediction, right_frame_previous_predicted_gesture, right_frame_confident_gesture, previous_right_frame_confident_gesture, right_frame_prediction_values, right_channels, right_frame_predicted_gesture_count)
        

        
    
    
# Sidebar
sidebar = tk.Frame(root, width=200, bg='lightgray')

info = tk.Frame(sidebar, bg='lightblue', height=100)

CurrentGesture = tk.Label(info, text="No Gesture Detected")
CurrentGesture.pack()

CurrentFile = tk.Label(info, text="No Audiofile Selected")
CurrentFile.pack()

currentMode = tk.Label(info, text="Current Mode: " + audioMode)
currentMode.pack()

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

left_file_button = Button(sidebar,
                        text="Select Left Hand Files",
                        command = select_file_left)
left_file_button.pack()

right_file_button = Button(sidebar,
                        text="Select Right Hand Files",
                        command = select_file_right)
right_file_button.pack()

sidebar.pack(side=tk.LEFT, fill=tk.Y)

# Gesture Icon Row
gestureIcons = tk.Frame(root, height=200)

imageNames = []
for dirroot, dirs, files in os.walk("GestureIcons"):
    for i, file in enumerate(files):
        imageNames.append(file)        #gestureIcon.grid_columnconfigure(i, weight=1)

images = []
imageText = []
for i in range(len(imageNames)):
    img = ImageTk.PhotoImage(Image.open(os.path.join("GestureIcons", imageNames[i])).resize((200,200)))
    images.append(img)
    Label(gestureIcons, image=img).grid(row=0, column=i)
    iconText = tk.Label(gestureIcons, text=list(gestures.values())[i])
    iconText.grid(row=1, column=i)
    imageText.append(iconText)

        #gestureIcon.pack()


gestureIcons.grid_columnconfigure(0, weight=1)

gestureIcons.pack(side=tk.BOTTOM)

img = Image.open("CameraDisabled.png")
img = img.resize((width, height))
img = ImageTk.PhotoImage(image=img)

webcamPanel = Label(root, image=img)
webcamPanel.pack()
root.mainloop()



        