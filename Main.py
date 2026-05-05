import os

import keyboard
from just_playback import Playback
import time
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import tkinter.font as tkFont
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
#import sounddevice as sd

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

stop_event = threading.Event()

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

global curTrackTime
curTrackTime = 0.0
trackLength = 0

audioMode = "None"
previousPredictedGesture = ""
predictedGesture = None
predictedGestureCount = 0
confidentGesture = ""
activated_Stems = []

def reset_system(event):
    global audioMode, previousPredictedGesture, predictedGesture, predictedGestureCount, confidentGesture, camera_open, depth_open, activated_Stems
    audioMode = "None"
    previousPredictedGesture = ""
    predictedGesture = None
    predictedGestureCount = 0
    confidentGesture = ""
    activated_Stems = []
    kinect.close()
    
    stop_event.set()
    camera_open = False
    #camera_update_thread.join()

    camera_open = False
    depth_open = False
    #gesture_recognition_thread.join()
    
    webcamPanel.config(image=img)
    
    CurrentFile['text'] = "No Audiofile Selected"
    CurrentGesture['text'] = "No Gesture Detected"
    gesture_recognition_button.config(state=tk.DISABLED)
    if(pygame.mixer.music.get_busy()):
        pygame.mixer.music.stop()


root = tk.Tk()
root.bind('<Escape>', reset_system)
root.geometry('1920x1080')
filename = "Initial"

label_widget = Label(root)
label_widget.pack()

def select_file():
    
    global audioMode, channels, curTrackTime, trackLength
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

        gesture_controls = ["Pause", "Volume Up", "Volume down", "", "", "Play", "", "+5 Seconds", "-5 Seconds", "Stop and Reset"]
        for i, gesture in enumerate(gesture_controls):
            stemText = tk.Label(gestureIcons, text=gesture)
            stemText.grid(row=2, column=i)

        curTrackTime = 0.0
        trackLength = pygame.mixer.Sound(filenames[0]).get_length()
        audioMode = "Single"
        music_file = filenames[0]
        #print(len(filenames))
        pause_button.config(state=tk.NORMAL)
        play_button.config(state=tk.DISABLED)
        #print(filenames[0])
        CurrentFile['text'] = "Single Audiofile Loaded"
        #playback.load_file(filename)
        #playback.play()
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
    
    elif 8 > len(filenames) > 1:
        audioMode = "Multi"
        stemText = tk.Label(gestureIcons, text="Build Stem")
        stemText.grid(row=2, column=0)
        for i, file in enumerate(filenames):
            stemText = tk.Label(gestureIcons, text=file.split("/")[-1])
            stemText.grid(row=2, column=i+1)
        pygame.mixer.set_num_channels(len(filenames))
        stems = [pygame.mixer.Sound(file) for file in filenames]
        CurrentFile['text'] = "Multiple Stems Loaded"
        channels = [pygame.mixer.Channel(i) for i in range(len(stems))]

        for i, channel in enumerate(channels):
            channel.play(stems[i])
            

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
    global camera_open, curTrackTime
    time.sleep(0.25)
    if(kinect.has_new_color_frame()):
        gesture_recognition_button.config(state=tk.NORMAL)

        camera_open = True
        stop_event.clear()
        close_camera_button.config(state=tk.NORMAL)
        global camera_update_thread
        camera_update_thread = threading.Thread(target=update_camera_loop)
        camera_update_thread.start()
    else:
        tk.messagebox.showerror(title="Camera Error", message="No Kinect detected, please ensure your Kinect is properly connected and restart the application.")
        

def update_camera_loop():
    if(stop_event.is_set()):
        return
    while camera_open == True:    
        update_camera()

def update_camera():
    if(camera_open == False):
        webcamPanel.config(image=img)

    #else:
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
        global gesture_recognition_thread
        gesture_recognition_thread = threading.Thread(target=gesture_recognition_loop)
        gesture_recognition_thread.start()

def gesture_recognition_loop():
    while camera_open == True:
        gesture_recognition()

def gesture_recognition():
    global previousPredictedGesture, predictedGesture, predictedGestureCount, confidentGesture, camera_open, curTrackTime, trackLength, activated_Stems
    #time.sleep(0.35)
    if (kinect.has_new_depth_frame() and camera_open == True):
        depth_frame = kinect.get_last_depth_frame()

        

        depth_frame = np.reshape(depth_frame, (424, 512))
        valid = depth_frame[depth_frame > 0]
        threshold = np.percentile(valid, 2) + 20


        depth_frame = np.array(depth_frame, dtype=np.float32)
        depth_frame = cv2.medianBlur(depth_frame.astype(np.uint16), 5)

        depth_frame[depth_frame > threshold] = 0
        #print ("Min depth value: ", depth_frame[depth_frame > 0].min())
        nonZeroPixels = cv2.countNonZero(depth_frame)
        if((nonZeroPixels < 3000) or (depth_frame[depth_frame > 0].min() > 800)):
            #print("No hand detected")
            predictedGesture = None
            CurrentGesture['text'] = "No hand detected"
            if(audioMode == "Multi"):
                activated_Stems = []
                for i, channel in enumerate(channels):
                    channel.set_volume(1)
                
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

            # TEST
            convertedDepthFrame = Image.fromarray(hand_crop)
            convertedDepthFrame = ImageTk.PhotoImage(image=convertedDepthFrame)
            
            cropped_depth_feed.config(image=convertedDepthFrame)
            cropped_depth_feed.image = convertedDepthFrame
            #END TEST


            model_prediction = model.predict(hand_crop.reshape(1, 128, 128, 1), verbose=0)
            predictedGestureValue = model_prediction.argmax()
            predictedGesture = gestures[model_prediction.argmax()]
            #print(predictedGesture)
            

        if(predictedGesture == previousPredictedGesture):
            if(predictedGestureCount >= 5):
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


                # For gestures that should only be triggered once, i.e playing, pausing, skipping through the track
                if(audioMode == "Single"):
                    if(confidentGesture != previousConfidentGesture):
                        match confidentGesture:
                            case "Fist":
                                pygame.mixer.music.pause()
                                pass
                            case "Five Fingers":
                                pygame.mixer.music.unpause()
                                pass
                            case "L":
                                #Needs redone, doesn't work
                                curPos = pygame.mixer.music.get_pos() / 1000
                                if (curTrackTime + curPos + 5 < trackLength):
                                    curTrackTime += curPos + 5
                                    pygame.mixer.music.set_pos(curTrackTime)
                                pass
                            
                            case "Thumbs Up":
                                curPos = pygame.mixer.music.get_pos() / 1000
                                if (curTrackTime - curPos - 5 > 0):
                                    curTrackTime -= curPos - 5
                                    pygame.mixer.music.set_pos(curTrackTime)
                                else:
                                    pygame.mixer.music.set_pos(0)
                                pass
                            
                            case "Rock On":
                                pygame.mixer.music.set_pos(0)
                                pygame.mixer.music.pause()
                                pass
                            case "Surfer":
                                #pygame.mixer.music.fadeout(2000)
                                #pygame.mixer.music.set_pos(0)
                                #pygame.mixer.music.pause()
                                pass
                    
                    # For gestures that should be continuously triggered, i.e volume control
                    else:
                        curVolume = pygame.mixer.music.get_volume()
                        match confidentGesture:
                            
                            case "Single Finger":
                                
                                pygame.mixer.music.set_volume(curVolume + 0.05)
                                pass
                            case "Two Fingers":
                                pygame.mixer.music.set_volume(curVolume - 0.05)
                                pass
                        curVolume = pygame.mixer.music.get_volume()
                        CurrentVolume['text'] = "Volume: " + str(round(curVolume,2))    

                elif (audioMode == "Multi"):

                    #print(confidentGesture, previousConfidentGesture)
                    if(confidentGesture != previousConfidentGesture):

                        curGestureIndex = list(gestures.values()).index(confidentGesture) - 1

                        # If previousconfident gesture is fist
                        # Get track, add it to 'active stems'
                        # Play all active stems, mute all others
                        if(previousConfidentGesture == "Fist"):
                            activated_Stems.append(curGestureIndex)
                            for i, channel in enumerate(channels):
                                if(i not in activated_Stems):
                                    channel.set_volume(0)
                                else:
                                    channel.set_volume(1)
                        

                        elif (confidentGesture != "Fist"):
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
        
    else:
        predictedGesture = None
        if(audioMode == "Multi"):
                activated_Stems = []
                for i, channel in enumerate(channels):
                    channel.set_volume(1)
        print("No depth frame")
        print("Camera Open: ", camera_open)
        
    
resize_font = tkFont.Font(family="Arial", size=15)
# Sidebar
sidebar = tk.Frame(root, width=200, bg='lightgray')

info = tk.Frame(sidebar, bg='lightblue', height=100)

CurrentGesture = tk.Label(info, text="No Gesture Detected", font=resize_font)
CurrentGesture.pack()

CurrentFile = tk.Label(info, text="No Audiofile Selected", font=resize_font)
CurrentFile.pack()

currentMode = tk.Label(info, text="Current Mode: " + audioMode, font=resize_font)
currentMode.pack()

getInitVolume = pygame.mixer.music.get_volume()
CurrentVolume = tk.Label(info, text="Volume: " + str(getInitVolume), font=resize_font)
CurrentVolume.pack()

info.pack(fill=tk.X)

open_button = Button(
    sidebar,
    text='Open a File',
    command=select_file,
    font=resize_font
)
open_button.pack()


pause_button = Button(
    sidebar,
    text='Pause',
    command=pause_command,
    state=tk.DISABLED,
    font=resize_font
)
pause_button.pack()

play_button = Button(
    sidebar,
    text='Play',
    command=play_command,
    state=tk.DISABLED,
    font=resize_font
)
play_button.pack()

feed_button = Button(sidebar, 
                     text="Open Camera", 
                     command=on_camera_button_click,
                     font=resize_font)
feed_button.pack()

close_camera_button = Button(sidebar,
                      text="Close Camera",
                      command = close_camera,
                      state=tk.DISABLED,
                      font=resize_font)
close_camera_button.pack()

gesture_recognition_button = Button(sidebar,
                        text="Start Gesture Recognition",
                        command = on_gesture_button_click,
                        state=tk.DISABLED,
                        font=resize_font)  
gesture_recognition_button.pack()

cropped_depth_feed = Label(sidebar,
                           image=None,
                           font=resize_font)
cropped_depth_feed.pack(side="bottom")

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
    img = ImageTk.PhotoImage(Image.open(os.path.join("GestureIcons", imageNames[i])).resize((150,150)))
    images.append(img)
    Label(gestureIcons, image=img, font=resize_font).grid(row=0, column=i)
    iconText = tk.Label(gestureIcons, text=list(gestures.values())[i], font=resize_font)
    iconText.grid(row=1, column=i)
    imageText.append(iconText)

        #gestureIcon.pack()


gestureIcons.grid_columnconfigure(0, weight=1)

gestureIcons.pack(side=tk.BOTTOM)

img = Image.open("Images/CameraDisabled.png")
img = img.resize((width, height))
img = ImageTk.PhotoImage(image=img)

webcamPanel = Label(root, image=img)
webcamPanel.pack()
root.mainloop()



        