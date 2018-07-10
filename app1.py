# coding:utf-8
import sys
import dlib
import cv2

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    img_path = 'demo.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    dets = detector(img, 1)
    print('faces: %d' % len(dets))

    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                     face.bottom()))

        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.namedWindow(img_path, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(img_path, img)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
