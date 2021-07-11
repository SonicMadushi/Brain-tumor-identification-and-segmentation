import cv2
import numpy as np

frame=cv2.imread('test_images/test (2).jpeg')

gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

ret,thresh=cv2.threshold(gray,168,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print('no. contours:',len(contours))

for cnt in contours:
    #cv2.drawContours(frame, [cnt], 0, (0,255,255), 3)
    area=cv2.contourArea(cnt)
    print(area)
    if(area>500):

        x,y,w,h = cv2.boundingRect(cnt)
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(frame, [cnt], 0, (0,0,255), 1)
        
cv2.imshow('frame',frame)
cv2.imshow('thresh',thresh)

k = cv2.waitKey(0)        
