import cv2
import numpy as np

name=input("enter your name:")

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

f_list=[]
while True:
    ret,frame=cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=classifier.detectMultiScale(gray,1.5,5)

    faces=sorted(face,key=lambda x:x[2]*x[3],reverse=True)

    face=faces[:1]

    if len(face)==1:
        face=face[0]
        x,y,w,h=face
        im_face=frame[y:y+h,x:x+w]
        cv2.imshow("face",im_face)

    if not ret:
        continue
    
    cv2.imshow("full",frame)
    
    key=cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('z'):
        if len(face) > 0 :
            gray_face=cv2.cvtColor(im_face,cv2.COLOR_BGR2GRAY)
            gray_face=cv2.resize(gray_face,(100,100))
            print(type(gray_face),gray_face.shape)
            gray_face.reshape(-1)
            f_list.append(gray_face)

        if len(f_list)==10:
            break
np.save(name,np.array(f_list))
cap.release()
cv2.destroyAllWindows()
