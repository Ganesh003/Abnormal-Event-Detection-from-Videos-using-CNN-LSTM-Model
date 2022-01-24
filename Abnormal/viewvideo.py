import cv2
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import os
from os.path import isfile, join


def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    print(files)
    r=[]
    for m in files:
    	print(m)
    	s=m.split(".")
    	print(s[0])
    	if s[0]!="Thumbs":
    		r.append(int(s[0]))
    print(r)   
    r1 = sorted(r)
    print("r1",r1) 
    files=[]
    for m in r1:
    	d=str(m)+".jpg"
    	print(d)
    	files.append(d)
    #for sorting the file names properly
    #files.sort(key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def viewVideo(x,y):
    cam=cv2.VideoCapture('out.avi')
    while True:
        rval,frame=cam.read()
        #print(rval)
        if rval==True:
            cv2.imshow("video",frame)
            if cv2.waitKey(1)==ord('q'):
                break
        else:
            break
    cam.release()
    cv2.destroyAllWindows()

    
    plt.plot(x, y) 
    plt.xlabel('Frame') 
    plt.ylabel('Value') 
    plt.title('Abnormal Detection!') 
    plt.pause(5)
    plt.show(block=False)
    plt.close()
    
    pathIn= 'Output/'
    pathOut = 'video.avi'
    fps = 10.0
    convert_frames_to_video(pathIn, pathOut, fps)

#x = [1,2,3] 
#y = [2,4,1] 
#viewVideo(x,y)


