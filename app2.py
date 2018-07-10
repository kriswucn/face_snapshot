# coding:utf-8
import cv2
import dlib
import datetime
import os
import numpy as np
import uuid

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()

    current_path = os.getcwd()
    predictor_name = 'shape_predictor_68_face_landmarks.dat'
    predictor_path = os.path.join(current_path, predictor_name)
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture('/home/kriswu/PycharmProjects/face_detection/10End.mp4')
    # cap = cv2.VideoCapture('rtsp://admin:admin12345@192.168.2.94/video/1')
    # 人脸截图路径
    save_path = os.path.join(current_path, 'faces')
    print(save_path)

    while cap.isOpened():
        ret, cv_img = cap.read()
        kk = cv2.waitKey(1)
        if cv_img is None:
            print('no face found.')
            break

        img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        rects = detector(img, 0)
        print('[%s] %d face(s) found.' % (datetime.datetime.now(), len(rects)))

        # 截取人脸
        if len(rects) != 0:
            for k, d in enumerate(rects):
                # 矩形大小
                pos_start = tuple([d.left(), d.top()])
                pos_end = tuple([d.right(), d.bottom()])
                height = d.bottom() - d.top()
                width = d.right() - d.left()

        # 根据人脸大小生成空白图片

            cv2.rectangle(cv_img, pos_start, pos_end, (0, 255, 255), 2)
            blank_img = np.zeros((height, width, 3), np.uint8)

        # 按下s保存图片
        # if kk == ord('s'):
            if 1 == 1:
                for x in range(height):
                    for y in range(width):
                        try:
                            blank_img[x][y] = cv_img[d.top() + x][d.left() + y]
                        except IndexError:
                            print('index is out of bounds...')
                            continue

                # img_name = str(uuid.uuid4()) + '.jpg'
                img_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + '.jpg'
                full_img_path = os.path.join(save_path, img_name)
                cv2.imwrite(full_img_path, blank_img)
                print(full_img_path)

            for idx, face in enumerate(rects):
                shape = predictor(img, face)

                for i, pt in enumerate(shape.parts()):
                    pt_pos = (pt.x, pt.y)
                    # cv2.circle(img, pt_pos, 2, (0, 0, 0), 1)
                    cv2.circle(img, pt_pos, 2, (0, 255, 0), -1)
            win.clear_overlay()
            win.set_image(img)
            # win.add_overlay(dets)

    cap.release()
