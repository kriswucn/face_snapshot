# coding:utf-8
import face_recognition
import datetime


def compare_face():
    kris_image = face_recognition.load_image_file("kris.jpg")
    male_image = face_recognition.load_image_file("male.jpg")
    unknown_image = face_recognition.load_image_file("unknown.jpg")

    kris_encoding = face_recognition.face_encodings(kris_image)[0]
    male_encoding = face_recognition.face_encodings(male_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([kris_encoding, male_encoding], unknown_encoding)
    labels = ['kris', 'male']

    print('results:' + str(results))

    for i in range(0, len(results)):
        if results[i]:
            print('The person is:' + labels[i])


if __name__ == '__main__':
    print(datetime.datetime.now())
    compare_face()
    print(datetime.datetime.now())
