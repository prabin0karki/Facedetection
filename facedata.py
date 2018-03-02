import cv2
import numpy as np
faceDetection=cv2.CascadeClassifier('/home/prabin/Downloads//haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0)
id=raw_input("Enter user id=")
name=raw_input("Enter user name=")
sampleNumber=0
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetection.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sampleNumber=sampleNumber+1
        user=name
        cv2.imwrite("dataset/"+str(name)+"."+str(id)+"."+str(sampleNumber)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
    cv2.imshow("Face",img)
    cv2.waitKey(100)
    if(sampleNumber>20):
        break
cam.release()