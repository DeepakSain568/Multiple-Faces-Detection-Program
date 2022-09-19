# Multiple-Faces-Detection-Program

from mtcnn import MTCNN
import cv2 as cv
detector = MTCNN()
cap = cv.VideoCapture(0)

while True:
    ret,Frame = cap.read()
    result = detector.detect_faces(Frame)
    for r in result:
        X,Y,height,width =r['box']
        cv.rectangle(Frame,(X,Y),(X+height,Y+width),(255,0,0),3)
        facial_features = ['left_eye','right_eye','nose','mouth_left','mouth_right']
        for i in facial_features:
            X_1,Y_1= r['keypoints'][i]
            cv.circle(Frame,center = (X_1,Y_1),radius=2,color=(255,0,0),thickness= 1)
            cv.imshow('result',Frame)
    if cv.waitKey(3) & 0XFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
