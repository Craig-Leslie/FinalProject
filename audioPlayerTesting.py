

from tkinter import *

import os
import tkinter as tk
from PIL import Image, ImageTk



root = tk.Tk()
gestureIcons = tk.Frame(root, bg='red', height=200)

imgs = []

for dirroot, dirs, files in os.walk("GestureIcons"):
    for i, file in enumerate(files):
        imgs.append(file)        #gestureIcon.grid_columnconfigure(i, weight=1)

img2 = []
for i in range(len(imgs)):
    img2.append(ImageTk.PhotoImage(Image.open(os.path.join("GestureIcons", imgs[i])).resize((100,100))))
    Label(gestureIcons, image=img2[i]).grid(row=0, column=i)
        #gestureIcon.pack()
gestureIcons.grid_columnconfigure(0, weight=1)
gestureIcons.pack(side=tk.BOTTOM, fill=tk.X)

root.mainloop()