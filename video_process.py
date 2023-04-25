import cv2
from cv2 import COLOR_BGR2GRAY
import numpy as np

video_path = 'E:datasets/yjsk_test/yjsk_videos1/0001_normal.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        gray = cv2.cvtColor(frame, COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        out.write(dst)
        # cv2.imshow('binary', dst)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()