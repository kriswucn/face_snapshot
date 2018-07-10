# coding；utf-8
import sys
import dlib
import cv2
import os

if __name__ == '__main__':
    current_path = os.getcwd()
    predictor_name = 'shape_predictor_68_face_landmarks.dat'
    predictor_path = os.path.join(current_path, predictor_name)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    image_file = os.path.join(current_path, 'demo.jpg')
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)

    b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])

    dets = detector(img, 1)
    print('%d face(s).' % len(dets))

    for idx, face in enumerate(dets):
        shape = predictor(img, face)

        for i, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (125, 2, 0), 1)

    cv2.namedWindow('aaa', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('aaa', img)

k = cv2.waitKey(0)
cv2.destroyAllWindows()
