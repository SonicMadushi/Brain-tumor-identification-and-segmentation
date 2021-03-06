#brainTumors creating dataset
import cv2,os

data_path='brain_tumor_dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict={} #empty dictionary

for i in range(len(categories)):
    label_dict[categories[i]]=labels[i]

print(label_dict)
print(categories)
print(labels)
img_size=50
dataset=[]

for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        #cv2.imshow('LIVE',img)
        #cv2.waitKey(100)
        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #Coverting the image into gray scale
            resized=cv2.resize(gray,(img_size,img_size))
            #resizing the gray scale into 50x50, since we need a fixed common size for all the images in the dataset
            dataset.append([resized,label_dict[category]])
            #appending the image and the label(categorized) into the list (dataset)
        except Exception as e:
            print(e)
            #if any exception rasied, the exception will be printed here. And pass to the next image

len(dataset)

from random import shuffle

shuffle(dataset)

data=[]
target=[]

for feature,label in dataset:
    data.append(feature)
    target.append(label)

import numpy as np

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)

from keras.utils import np_utils

new_target=np_utils.to_categorical(target)

np.save('data',data)
np.save('target',new_target)


