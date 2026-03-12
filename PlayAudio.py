#import keyboard
from just_playback import Playback
import time
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
from PIL import Image, ImageTk




vid = cv2.VideoCapture(0)


width, height = 800, 600

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

    _, frame = vid.read()

    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)

    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(10, open_camera)

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

canvas = tk.Canvas(root)

feed_button.pack()

open_button.pack(expand=True)

pause_button.pack()
play_button.pack()
canvas.pack()
label = tk.Label(root, text="Audio Path")
label.pack()
print(filename)
root.mainloop()



        