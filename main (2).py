import numpy as np
import cv2
import imutils
import random

import warnings
warnings.filterwarnings("ignore")


feature=[]
label=[]
image_count=0
t=0
import os
path=os.path.join(os.getcwd(),'brain_tumor_dataset')
r=128
up_down_flip=np.zeros((r,r))
mirror=np.zeros((r,r))
black=3
white=255
snr=1.2
noise=np.zeros((r,r))
for (root,dirs,files) in os.walk(path):
    if files !=[]:
        l=len(files)
        
        for i in range(0,l):
            
            path=os.path.join(root,files[i])
            
            
            full_size_image = cv2.imread(path,0)
            cv2.imshow('full_size_image',full_size_image)
            cv2.waitKey(1)
            
            resized_image=cv2.resize(full_size_image, (r,r), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('resized_image',resized_image)
            cv2.waitKey(1)
            feature.append(resized_image)
            label.append(image_count)

            for i in range(0,r):
                up_down_flip[r-i-1,:]=resized_image[i,:]
            cv2.imshow('up_down_flip',np.uint8(up_down_flip))
            cv2.waitKey(1)
            feature.append(up_down_flip)
            label.append(image_count)

            

            for i in range(0,r):
                mirror[:,r-i-1]=resized_image[:,i]
            cv2.imshow('mirror',np.uint8(mirror))
            cv2.waitKey(1)
            feature.append(mirror)
            label.append(image_count)

            sn_img=mirror

            number_of_pixels = random.randint(0, 50)
            
            for i in range(number_of_pixels):
                
                
                y_coord=random.randint(0, r - 1)
                  
               
                x_coord=random.randint(0, r - 1)
                  
               
                sn_img[y_coord][x_coord] = 255
                  
           
            number_of_pixels = random.randint(0 , 50)
            for i in range(number_of_pixels):
                
                
                y_coord=random.randint(0, r - 1)
                  
                
                x_coord=random.randint(0, r - 1)
                  
               
                sn_img[y_coord][x_coord] = 0

            cv2.imshow('salt and pepper',np.uint8(sn_img))
            cv2.waitKey(1)
            feature.append(sn_img)
            label.append(image_count)

            Rotated_image = imutils.rotate(resized_image, angle=45)
            cv2.imshow('rotated',np.uint8(Rotated_image))
            cv2.waitKey(1)
            feature.append(Rotated_image)

            label.append(image_count)
            t=t+5
        image_count=image_count+1

feature=np.asarray(feature)
feature = feature.reshape(t,r,r,1).astype('float32')
feature=feature/255
label=np.asarray(label).astype('uint8')

from keras.utils import np_utils
label = np_utils.to_categorical(label)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(feature, label, test_size=0.3, random_state=42)


num_classes = ytest.shape[1]

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers


def baseline_model():

    model = Sequential()
    model.add(Conv2D(64, (10, 10),input_shape=(r, r, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(num_classes, activation='softmax'))
	
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
loss,acc = model.evaluate(xtest, ytest, verbose=2)
print("CNN Accuracy:", acc*100)

    
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


    
