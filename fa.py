import cv2
import numpy as np
faceDetection=cv2.CascadeClassifier('/home/prabin/Downloads/haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0)
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetection.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()