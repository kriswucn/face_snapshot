# coding:utf-8
import cv2
import dlib
import datetime
import sys
import os
import numpy as np
import face_recognition
import time


class FaceSnapshot(object):
    def __init__(self, video_path, save_path):
        self.video_path = video_path
        self.save_path = save_path
        self.detector = dlib.get_frontal_face_detector()
        self.win = dlib.image_window()
        self.predictor = None
        self.cap = None
        # 人脸库
        self.face_lib = []
        self.person_names = []

    # 加载预测器
    def load_predictor(self):
        current_path = os.getcwd()
        # predictor_name = 'shape_predictor_68_face_landmarks.dat'
        predictor_name = 'shape_predictor_5_face_landmarks.dat'
        predictor_path = os.path.join(current_path, predictor_name)
        self.predictor = dlib.shape_predictor(predictor_path)

    # 加载人脸库
    def load_face_lib(self, face_lib_folder='face_lib'):
        current_folder = os.getcwd()
        full_face_lib_folder = os.path.join(current_folder, face_lib_folder)
        subs = os.listdir(full_face_lib_folder)
        # 人脸图片
        img_files = []

        for i in subs:
            full_img = os.path.join(full_face_lib_folder, i)
            if os.path.isfile(full_img):
                person_name = i.split('.')[0]
                self.person_names.append(person_name)
                img_files.append(full_img)
                tmp_arr_img = face_recognition.load_image_file(full_img)
                tmp_128dim = face_recognition.face_encodings(tmp_arr_img)[0]
                self.face_lib.append(tmp_128dim)

    # 人脸比对
    def compare_face(self, xface):
        current_folder = os.getcwd()
        full_xface = os.path.join(current_folder, xface)

        x_arr_img = face_recognition.load_image_file(full_xface)
        try:
            x_128dim = face_recognition.face_encodings(x_arr_img)[0]
        except IndexError:
            # print('x_128dim is out of bound')
            return

        result = face_recognition.compare_faces(self.face_lib, x_128dim, tolerance=0.3)

        matched = False
        for i in range(len(result)):
            if result[i]:
                print('%s was found.' % self.person_names[i])
                matched = True

        if matched is False:
            print('Stranger')

    # 人脸检测
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
            val = frame_counter % 30
            frame_counter += 1

            if val != 0:
                continue

            img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            if img is None:
                print('Video file maybe broken.')
                sys.exit(-1)

            rects = self.detector(img, 0)

            if len(rects) == 0:
                # print('[%s] .' % datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
                print('.')
            # print('[%s] %d face(s) found.' % (datetime.datetime.now(), len(rects)))
            self.snapshot_face(rects, cv_img)
            # 绘制68个点
            self.draw_landmarks(rects, img)
            # 窗口显示图片
            self.win.clear_overlay()
            self.win.set_image(img)

    # 人脸截图
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
            # 人脸比对
            self.compare_face(full_img_path)
            print(full_img_path)

    def draw_landmarks(self, rects, img):
        for idx, face in enumerate(rects):
            shape = self.predictor(img, face)

            for i, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                # cv2.circle(img, pt_pos, 2, (0, 0, 0), 1)
                cv2.circle(img, pt_pos, 2, (0, 255, 0), -1)


if __name__ == '__main__':
    # video_path1 = 'v/2.mp4'
    video_path1 = 'rtsp://admin:admin12345@192.168.2.94/video/1'
    # video_path1 = 'v/10End.mp4'
    save_path1 = 'snap_faces'
    face_snapshot = FaceSnapshot(video_path1, save_path1)
    face_snapshot.load_predictor()
    face_snapshot.load_face_lib()
    face_snapshot.detect_face()
