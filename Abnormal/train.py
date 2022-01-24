from model import load_model
import numpy as np 
import argparse


def training():
	model=load_model()
	print('Model has been loaded')
	X_train=np.load('training.npy')
	frames=X_train.shape[2]
	#Need to make number of frames divisible by 10


	frames=frames-frames%10

	X_train=X_train[:,:,:frames]
	X_train=X_train.reshape(-1,227,227,10)
	X_train=np.expand_dims(X_train,axis=4)
	Y_train=X_train.copy()

	epochs=3
	batch_size=1
	model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs)
	model.save('AnomalyDetector.h5')
	print("Training Finished")




