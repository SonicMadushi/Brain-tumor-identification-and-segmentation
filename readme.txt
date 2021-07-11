#“BRAINCARE” - AN ARTIFICIAL INTELLIGENCE BASED SYSTEM FOR BRAIN TUMOR DETECTION 
# 2020-028
#IT16076662 H.T.M.D Samaranayake 

#Abstrat
A brain tumor in people of all ages in the most severe disease leading to a severe disease leading to very low life expectancy. Brain tumor misdiagnosis will result in an incorrect medical collision, which will limit patient chances. Tumor area is  The accurate identifying and segmenting of the tumor of the brain tumor is a key point to make proper treatment planning to cure and improve the existing.  Precise segmentation of the tumor and identification is a prerequisite and compelling Move Next Step a Machine-assisted brain tumor analysis and Operative planning. Subjectivity Identifying a brain tumor and 
Broadly accepted segmentations in clinical diagnosis and treatment. An automatic and Objective brain tumor system identification and Segmentation is expected to be strong. However, Some are still facing 	contests such as More accuracy in segmentation, requiring a priori awareness or necessitating Human Response. Identifying a brain tumor and segmentation from The  diagnosis, control, and treatment preparation of the disease includes magnetic resonance imagi ng (MRIs). Handbook delineation practice requires Anatomical know-how, which is expensive time consuming and can be inaccurate due to human error. 
This uses a deep learning model to predict a brain tumor whether to identify a brain tumor or not. After gathering, the results to identify the tumor shape and size based on the segmentation of the brain tumor .Unsupervised deep learning can detect and remove the noise of the images, which makes the identification on brain tumors more accurate. A semantic segmentation and Identification for brain tumors from MRIs is stated. Data sets are trained repeatedly with different techniques to make an accurate identification and segmentation of the brain tumor.  Increasing the network depth improves the results. 
 

# Main Objective
The aim of our project is to develop a web platform with identifying a brain tumor and segmentate with a higher accuracy using image processing, neural networks. Assisting radiologists with whole staff with much time needed to treatment and saving the time for patient’s life.  Using a higher accuracy prediction to identify the brain tumor and segment it through MRIs.   

# specific Objectives

•	To take input as a MRI image and through that image we are design to identify a brain tumor at any stage (even in the smallest stage ).  
•	To make it easy to read the MRI images for Radiologists which are difficult and take too much time to read the MRI images in present days 
•	Building a method to identify the growth rate of brain tumor which is more accurate than existing methods. 
•	Predict the side effects that can be happens due to the brain tumor and the threats that Causes to human body because of those side effects. 


#libraries

##H.T.M.D Samaranayake
#1)brain tumor identification 
import numpy as np
import cv2,os
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D

## brain tumor segmentation
import cv2
import os
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K

##frontend
from flask import Flask,render_template,request
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input
import keras.backend as K
import numpy as np




#How to run
Go to the workspace 
Type cmd.exe on folder path
In that folder path which displays in cmd.exe type python main.py
At the end go to the Running on 'http://127.0.0.1:5000/' you can go to the web app
