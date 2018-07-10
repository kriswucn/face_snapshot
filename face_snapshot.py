# coding:utf-8
import cv2
import dlib
import datetime
import sys
import os
import numpy as np


class FaceSnapshot(object):
    def __init__(self, video_path, save_path):
        self.video_path = video_path
        self.save_path = save_path
        self.detector = dlib.get_frontal_face_detector()
        self.win = dlib.image_window()
        self.predictor = None
        self.cap = None

    def get_predictor(self):
        current_path = os.getcwd()
        predictor_name = 'shape_predictor_68_face_landmarks.dat'
        predictor_path = os.path.join(current_path, predictor_name)
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_face(self):
        try:
            os.path.exists(self.video_path)
        except FileExistsError:
            print('Video file is invalid.')
            sys.exit(-1)

        self.cap = cv2.VideoCapture(self.video_path)

        while self.cap.isOpened():
            ret, cv_img = self.cap.read()
            img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

            if img is None:
                print('Video file maybe broken.')
                sys.exit(-1)

            rects = self.detector(img, 0)
            print('[%s] %d face(s) found.' % (datetime.datetime.now(), len(rects)))
