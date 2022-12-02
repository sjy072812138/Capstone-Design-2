# 1 加载库
import cv2
import dlib
# 2 打开摄像头
capture = cv2.VideoCapture(0)

# 3 获取人脸检测器
detector = dlib.get_frontal_face_detector()

# 4 获取人脸关键点检测模型
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = capture.read() # 读取视频流
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度转换
    faces = detector(gray, 1)# 人脸检测 1表示把图片放大一倍
    for face in faces: # 绘制每张人脸的矩形框和关键点
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 3) # 绘制矩形框
        shape = predictor(gray, face) # 识别检测关键点
        for pt in shape.parts(): # 获取关键点坐标
            pt_position = (pt.x, pt.y) # 每个点的坐标
            cv2.circle(frame, pt_position, 3, (255, 0, 0), -1) # 绘制关键点
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.imshow("face detection landmark", frame) # 显示效果
capture.release()
cv2.destroyAllWindows()
