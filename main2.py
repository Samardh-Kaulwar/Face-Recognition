import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (200,255,300), 2)
        eyes_gray = gray[y:y+h, x:x+w]
        eyes_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(eyes_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(eyes_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
