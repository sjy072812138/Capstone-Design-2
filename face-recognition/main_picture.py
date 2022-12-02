import cv2
import dlib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def face_recognition(pic_path='./pic/3.jpg'):
    frame = cv2.resize(cv2.imread(pic_path), (600, 800))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    for face in faces:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        shape = predictor(gray, face)
        pts = shape.parts()
        for pt in pts:
            pt_position = (pt.x, pt.y)
            cv2.circle(frame, pt_position, 2, (0, 0, 255), -1)  # 绘制关键点
        cv2.putText(frame, 'SunJiYao', (face.left(), face.top() - 10), None, 1, (0, 255, 0), 2)

    left_eye_width = max(pts[36].x, pts[37].x, pts[38].x, pts[39].x, pts[40].x, pts[41].x) - \
                     min(pts[36].x, pts[37].x, pts[38].x, pts[39].x, pts[40].x, pts[41].x)

    left_eye_height = max(pts[36].y, pts[37].y, pts[38].y, pts[39].y, pts[40].y, pts[41].y) - \
                     min(pts[36].y, pts[37].y, pts[38].y, pts[39].y, pts[40].y, pts[41].y)

    right_eye_width = max(pts[42].x, pts[43].x, pts[44].x, pts[45].x, pts[46].x, pts[47].x) - \
                     min(pts[42].x, pts[43].x, pts[44].x, pts[45].x, pts[46].x, pts[47].x)

    right_eye_height = max(pts[42].y, pts[43].y, pts[44].y, pts[45].y, pts[46].y, pts[47].y) - \
                     min(pts[42].y, pts[43].y, pts[44].y, pts[45].y, pts[46].y, pts[47].y)


    nose_height = max([pts[i].y for i in [27, 28, 29, 30, 31, 32, 33, 34, 35]]) - min([pts[i].y for i in [27, 28, 29, 30, 31, 32, 33, 34, 35]])
    nose_width = max([pts[i].x for i in [31, 32, 33, 34, 35]]) - min([pts[i].x for i in [31, 32, 33, 34, 35]])

    mouse_width = pts[54].x - pts[48].x
    mouse_height = pts[57].y - pts[51].y

    face_width = pts[16].x - pts[0].x
    face_height = pts[8].y - pts[19].y

    return frame

frame, fv = face_recognition(pic_path='./pic/SunJiYao/SunJiYao3.jpg')
frame, fv = face_recognition(pic_path='C:/Users/Administrator/Desktop/face-recognition/pic/SunJiYao/SunJiYao1.jpg')
