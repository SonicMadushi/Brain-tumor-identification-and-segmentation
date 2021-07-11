import cv2
import os
import numpy as np

original_path='original'
edited_path='edited'

edited_image_names=os.listdir(edited_path)


data_original=[]
data_edited=[]

img_size=256

for edited_image_name in edited_image_names:
    
    edited_image=cv2.imread(os.path.join(edited_path,edited_image_name))
    original_image_name='0 ('+edited_image_name[:-4]+').jpg'
    original_image=cv2.imread(os.path.join(original_path,original_image_name))
    
    edited_image=cv2.cvtColor(edited_image,cv2.COLOR_BGR2GRAY)
    original_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    
    edited_image=cv2.resize(edited_image,(img_size,img_size))
    original_image=cv2.resize(original_image,(img_size,img_size))
    
    data_edited.append(edited_image)
    data_original.append(original_image)

    data_original=np.array(data_original)/255
data_edited=np.array(data_edited)/255

data_original=data_original.reshape(data_original.shape[0],img_size,img_size,1)
data_edited=data_edited.reshape(data_edited.shape[0],img_size,img_size,1)

from matplotlib import pyplot as plt

plt.imshow(data_original[0].reshape(img_size,img_size),cmap='gray')

plt.imshow(data_edited[0].reshape(img_size,img_size),cmap='gray')

from sklearn.model_selection import train_test_split

train_data_original,test_data_original,train_data_edited,test_data_edited=train_test_split(data_original,data_edited,test_size=0.1)

plt.imshow(test_data_edited[0].reshape(img_size,img_size),cmap='gray')

from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(img_size, img_size, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history=autoencoder.fit(train_data_original, train_data_edited,epochs=100,validation_split=0.1,)

from matplotlib import pyplot as plt

plt.plot(history.history['loss'],'r',label='loss')
plt.plot(history.history['val_loss'],'b',label='loss')

autoencoder.save_weights('autoencorders_noise_removal.h5')

results=autoencoder.predict(train_data_original)

for i in range(len(results)):
    plt.imshow(results[i].reshape(img_size,img_size),cmap='gray')
    plt.savefig('results/'+str(i)+'-extracted-Autoencoder.png')
    plt.close()
    
    plt.imshow(train_data_edited[i].reshape(img_size,img_size),cmap='gray')
    plt.savefig('results/'+str(i)+'-extracted-original.png')
    plt.close()
    
    plt.imshow(train_data_original[i].reshape(img_size,img_size),cmap='gray')
    plt.savefig('results/'+str(i)+'-original.png')
    plt.close()
