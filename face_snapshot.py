# coding:utf-8
import cv2
import dlib
import datetime
import sys
import os
import numpy as np
import logging


class FaceSnapshot(object):
    def __init__(self, video_path, save_path):
        self.video_path = video_path
        self.save_path = save_path
        self.detector = dlib.get_frontal_face_detector()
        self.win = dlib.image_window()
        self.predictor = None
        self.cap = None
        self.logger = logging.getLogger(__name__)

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

        # 帧计数器
        frame_counter = 0

        while self.cap.isOpened():
            ret, cv_img = self.cap.read()

            if ret is False:
                break

            # 每1帧跳6帧
            val = frame_counter % 5
            frame_counter += 1

            if val != 0:
                continue

            img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            if img is None:
                print('Video file maybe broken.')
                sys.exit(-1)

            rects = self.detector(img, 0)

            if len(rects) == 0:
                print('.')

            # print('[%s] %d face(s) found.' % (datetime.datetime.now(), len(rects)))
            self.snapshot_face(rects, cv_img)
            # 绘制68个点
            self.draw_landmarks(rects, img)
            # 窗口显示图片
            self.win.clear_overlay()
            self.win.set_image(img)

    def snapshot_face(self, rects, cv_img):
        # 人脸区域
        if len(rects) != 0:
            for k, d in enumerate(rects):
                p_start = (d.left(), d.top())
                p_end = (d.right(), d.bottom())
                height = d.bottom() - d.top()
                width = d.right() - d.left()

            # 根据人脸大小生成临时图片

            cv2.rectangle(cv_img, p_start, p_end, (0, 255, 255), 2)
            tmp_img = np.zeros((height, width, 3), np.uint8)

            # 保存图片
            for x in range(height):
                for y in range(width):
                    try:
                        tmp_img[x][y] = cv_img[d.top() + x][d.left() + y]
                    except IndexError:
                        print('Index is out of bound...')

            img_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + '.jpg'
            full_img_path = os.path.join(self.save_path, img_name)
            cv2.imwrite(full_img_path, tmp_img)
            print(full_img_path)

    def draw_landmarks(self, rects, img):
        for idx, face in enumerate(rects):
            shape = self.predictor(img, face)

            for i, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                # cv2.circle(img, pt_pos, 2, (0, 0, 0), 1)
                cv2.circle(img, pt_pos, 2, (0, 255, 0), -1)


if __name__ == '__main__':
    video_path1 = 'v/bp10.mp4'
    save_path1 = 'snap_faces'
    face_snapshot = FaceSnapshot(video_path1, save_path1)
    face_snapshot.get_predictor()
    face_snapshot.detect_face()
