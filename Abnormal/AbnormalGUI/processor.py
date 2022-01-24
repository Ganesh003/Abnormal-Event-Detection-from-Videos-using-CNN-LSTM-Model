from keras.preprocessing.image import img_to_array,load_img
from sklearn.preprocessing import StandardScaler
import numpy as np 
import os 
import cv2
from scipy.misc import imresize 

imagestore=[]


vicap="";

def store(image_path):
	global imagestore
	img=load_img(image_path)
	img=img_to_array(img)
	#Resize the Image to (227,227,3) for the network to be able to process it. 
	img=imresize(img,(227,227,3))
	#Convert the Image to Grayscale
	gray=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]
	imagestore.append(gray)


def loadvideo(path):
	global vidcap,imagestore
	video=path
	cam=cv2.VideoCapture(path)
	while True:
		rval,frame=cam.read()
		print(rval)
		if rval==True:
			cv2.imshow("video",frame)
			if cv2.waitKey(1)==ord('q'):
				break
		else:
			break
	cam.release()
	cv2.destroyAllWindows()
	n=1
	sec=0
	vidcap=cv2.VideoCapture(path)
	success = getFrame(sec,n)
	while success:
		n=n+1
		sec = sec + 1.0
		sec = round(sec, 2)
		success = getFrame(sec,n)
	framepath='frames'
	images=os.listdir(framepath)
	for image in images:
		image_path=framepath+ '/'+ image
		store(image_path)

	imagestore=np.array(imagestore)
	print(imagestore.shape)
	a,b,c=imagestore.shape
	#Reshape to (227,227,batch_size)
	imagestore.resize(b,c,a)
	#Normalize
	imagestore=(imagestore-imagestore.mean())/(imagestore.std())
	#Clip negative Values
	imagestore=np.clip(imagestore,0,1)
	np.save('training.npy',imagestore)


def getFrame(sec,n):
	global vidcap
	vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
	hasFrames,frame = vidcap.read()
	if hasFrames:
		cv2.imwrite("frames/"+str(n)+".jpg", frame)
	return hasFrames
