import numpy as np
import cv2

cap = cv2.VideoCapture('kris.mp4')

if __name__ == '__main__':
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
