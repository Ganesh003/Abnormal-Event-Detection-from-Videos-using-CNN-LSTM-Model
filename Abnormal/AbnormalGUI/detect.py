import cv2
from model import load_model
import numpy as np 
from scipy.misc import imresize
#from test import mean_squared_loss
from keras.models import load_model
import viewvideo as vid



font = cv2.FONT_HERSHEY_SIMPLEX
x=[]
y=[]
threshold=0.0003498
frame_array = []
size=""

def mean_squared_loss(x1,x2):
	diff=x1-x2
	a,b,c,d,e=diff.shape
	n_samples=a*b*c*d*e
	sq_diff=diff**2
	Sum=sq_diff.sum()
	dist=np.sqrt(Sum)
	mean_dist=dist/n_samples

	return mean_dist
def getFrame(sec,n):
        global vidcap,x,y,size,threshold,frame_array,model
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,frame = vidcap.read()
        if hasFrames:
                imagedump=[]
                frame=imresize(frame,(227,227,3))
                gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                gray=(gray-gray.mean())/gray.std()
                gray=np.clip(gray,0,1)
                imagedump.append(gray)
                cv2.imwrite("output/"+str(n)+".jpg", frame)
                imagedump=np.array(imagedump)
                imagedump.resize(227,227,10)
                imagedump=np.expand_dims(imagedump,axis=0)
                imagedump=np.expand_dims(imagedump,axis=4)
                print('Processing data')
                print("n value",n)
                x.append(n)
                img = cv2.imread("output/"+str(n)+".jpg")
                height, width, layers = img.shape
                size = (width,height)
                output=model.predict(imagedump)
                loss=mean_squared_loss(imagedump,output)
                print(loss)
                #loss=0.0003444
                y.append(loss)
                if loss<threshold:
                        print('Anomalies Detected')
                        cv2.putText(img,'AnomalyDetector',(0,20), font, 0.5,(255,255,255),2,cv2.LINE_AA)
                        cv2.imwrite("output/"+str(n)+".jpg", img)
                frame_array.append(img)
        return hasFrames
    
    
def detect(path):
        global vidcap,x,y,size,threshold,frame_array,model
        modelpath='AnomalyDetector.h5'
        print('Loading model')
        model=load_model(modelpath)
        print('Model loaded')
        vidcap = cv2.VideoCapture(path)
        n=1
        sec=0
        success = getFrame(sec,n)
        while success:
                n=n+1
                sec = sec + 1.0
                sec = round(sec, 2)
                success = getFrame(sec,n)
        out = cv2.VideoWriter("out.avi",cv2.VideoWriter_fourcc(*'MPEG'), 1, (227,227))
        for i in range(len(frame_array)):
                #print(frame_array[i])
                out.write(frame_array[i])
        out.release()
        print(x)
        print(y)
        vid.viewVideo(x,y)
