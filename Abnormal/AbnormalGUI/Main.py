import tkinter as tk
from tkinter import Message ,Text
from tkinter import filedialog
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import processor as loadimg
import train as train
import detect as dete


from matplotlib import pyplot as plt



from_date = datetime.datetime.today()
currentDate = time.strftime("%d_%m_%y")
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale=1
fontColor=(255,255,255)
cond=0


window = tk.Tk()
window.title("Abnormal Event Detection")

 
window.geometry('1280x720')
window.configure(background='blue')
#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message1 = tk.Label(window, text="Abnormal Event Detection" ,bg="blue"  ,fg="white"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
message1.place(x=100, y=20)

lbl = tk.Label(window, text="Video Path",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
lbl.place(x=100, y=200)

#txt = tk.Entry(window,width=20,bg="yellow" ,fg="red",font=('times', 15, ' bold '))
#txt.place(x=400, y=215)

message2 = tk.Label(window, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message2.place(x=400, y=215)

lbl4 = tk.Label(window, text="Notification : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl4.place(x=100, y=500)

message = tk.Label(window, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=400, y=500)

path=""
def clear():
	txt.delete(0, 'end')    
	res = ""
	message.configure(text= res)
    
def submit():
        global path
        print("Submit")
        path = filedialog.askopenfilename()
        print(path)
        message2.configure(text= path)
        if path != "" :
                loadimg.loadvideo(path)
        else:
                res = "Laod Image"
                message.configure(text= res)
		
	
def Train():
	print("Train")
	train.training()
	res = "Train Finished"
	message.configure(text= res)
	#pred.Predict()
	
def Detect():
	print("Detect")
	global path
	print(path)
	dete.detect(path)
	res = "Detect Finished"
	message.configure(text= res)
	

    
    

addst = tk.Button(window, text="Load Video", command=submit  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
addst.place(x=100, y=600)

trainImg = tk.Button(window, text="Detect", command=Detect  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=700, y=600)

detect = tk.Button(window, text="Train", command=Train  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
detect.place(x=400, y=600)

quitWindow = tk.Button(window, text="QUIT", command=window.destroy  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1000, y=600)

clearButton = tk.Button(window, text="Clear", command=clear  ,fg="white"  ,bg="blue"  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)

 
window.mainloop()
