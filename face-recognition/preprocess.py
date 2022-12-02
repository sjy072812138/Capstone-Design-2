import numpy as np
import copy
import pymysql
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def face_recognition(pic_path='./pic/3.jpg'):
    frame = cv2.resize(cv2.imread(pic_path), (600, 800))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度转换
    faces = detector(gray, 1)  # 人脸检测 1表示把图片放大一倍
    for face in faces:  # 绘制每张人脸的矩形框和关键点
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)  # 绘制矩形框
        shape = predictor(gray, face)  # 识别检测关键点
        pts = shape.parts()
        for pt in pts:  # 获取关键点坐标
            pt_position = (pt.x, pt.y)  # 每个点的坐标
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

    return frame, [face_width, face_height, mouse_width, mouse_height,
                   nose_width, nose_height, left_eye_width, left_eye_height, right_eye_width, right_eye_height]


def face_recognition_name(pic_path='./pic/3.jpg', feature_id = ['unknown'], feature = [[0] * 10]):
    frame = cv2.resize(cv2.imread(pic_path), (600, 800))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度转换
    faces = detector(gray, 1)  # 人脸检测 1表示把图片放大一倍
    if len(faces) > 0:
        face = faces[0]
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)  # 绘制矩形框
        shape = predictor(gray, face)  # 识别检测关键点
        pts = shape.parts()
        for pt in pts:  # 获取关键点坐标
            pt_position = (pt.x, pt.y)  # 每个点的坐标
            cv2.circle(frame, pt_position, 2, (0, 0, 255), -1)  # 绘制关键点

        left_eye_width = max(pts[36].x, pts[37].x, pts[38].x, pts[39].x, pts[40].x, pts[41].x) - \
                         min(pts[36].x, pts[37].x, pts[38].x, pts[39].x, pts[40].x, pts[41].x)

        left_eye_height = max(pts[36].y, pts[37].y, pts[38].y, pts[39].y, pts[40].y, pts[41].y) - \
                          min(pts[36].y, pts[37].y, pts[38].y, pts[39].y, pts[40].y, pts[41].y)

        right_eye_width = max(pts[42].x, pts[43].x, pts[44].x, pts[45].x, pts[46].x, pts[47].x) - \
                          min(pts[42].x, pts[43].x, pts[44].x, pts[45].x, pts[46].x, pts[47].x)

        right_eye_height = max(pts[42].y, pts[43].y, pts[44].y, pts[45].y, pts[46].y, pts[47].y) - \
                           min(pts[42].y, pts[43].y, pts[44].y, pts[45].y, pts[46].y, pts[47].y)

        nose_height = max([pts[i].y for i in [27, 28, 29, 30, 31, 32, 33, 34, 35]]) - min(
            [pts[i].y for i in [27, 28, 29, 30, 31, 32, 33, 34, 35]])
        nose_width = max([pts[i].x for i in [31, 32, 33, 34, 35]]) - min([pts[i].x for i in [31, 32, 33, 34, 35]])

        mouse_width = pts[54].x - pts[48].x
        mouse_height = pts[57].y - pts[51].y

        face_width = pts[16].x - pts[0].x
        face_height = pts[8].y - pts[19].y

        this_face_feature = [face_width, face_height, mouse_width, mouse_height,
                   nose_width, nose_height, left_eye_width, left_eye_height, right_eye_width, right_eye_height]

        print("this_face_feature:", this_face_feature)
        print(feature_id)
        print(feature)
        feature_id.append('UnKnown')
        feature.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        print(feature_id)
        print(feature)
        min_diff = np.inf
        name = 'UnKnown'
        for i in range(len(feature_id)):
            dist = np.linalg.norm(np.array(feature[i]) - np.array(this_face_feature))
            print('dist: ', dist)
            if dist < min_diff:
                min_diff = dist
                name = feature_id[i]
        cv2.putText(frame, name, (face.left(), face.top() - 10), None, 1, (0, 255, 0), 2)
    else:
        pass
    return frame


def face_recognition_name_from_frame(frame, feature_id = ['unknown'], feature = [[0] * 10]):
    frame = cv2.resize(frame, (600, 800))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度转换
    faces = detector(gray, 1)  # 人脸检测 1表示把图片放大一倍
    if len(faces) > 0:
        face = faces[0]
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)  # 绘制矩形框
        shape = predictor(gray, face)  # 识别检测关键点
        pts = shape.parts()
        for pt in pts:  # 获取关键点坐标
            pt_position = (pt.x, pt.y)  # 每个点的坐标
            cv2.circle(frame, pt_position, 2, (0, 0, 255), -1)  # 绘制关键点

        left_eye_width = max(pts[36].x, pts[37].x, pts[38].x, pts[39].x, pts[40].x, pts[41].x) - \
                         min(pts[36].x, pts[37].x, pts[38].x, pts[39].x, pts[40].x, pts[41].x)

        left_eye_height = max(pts[36].y, pts[37].y, pts[38].y, pts[39].y, pts[40].y, pts[41].y) - \
                          min(pts[36].y, pts[37].y, pts[38].y, pts[39].y, pts[40].y, pts[41].y)

        right_eye_width = max(pts[42].x, pts[43].x, pts[44].x, pts[45].x, pts[46].x, pts[47].x) - \
                          min(pts[42].x, pts[43].x, pts[44].x, pts[45].x, pts[46].x, pts[47].x)

        right_eye_height = max(pts[42].y, pts[43].y, pts[44].y, pts[45].y, pts[46].y, pts[47].y) - \
                           min(pts[42].y, pts[43].y, pts[44].y, pts[45].y, pts[46].y, pts[47].y)

        nose_height = max([pts[i].y for i in [27, 28, 29, 30, 31, 32, 33, 34, 35]]) - min(
            [pts[i].y for i in [27, 28, 29, 30, 31, 32, 33, 34, 35]])
        nose_width = max([pts[i].x for i in [31, 32, 33, 34, 35]]) - min([pts[i].x for i in [31, 32, 33, 34, 35]])

        mouse_width = pts[54].x - pts[48].x
        mouse_height = pts[57].y - pts[51].y

        face_width = pts[16].x - pts[0].x
        face_height = pts[8].y - pts[19].y

        this_face_feature = [face_width, face_height, mouse_width, mouse_height,
                   nose_width, nose_height, left_eye_width, left_eye_height, right_eye_width, right_eye_height]

        print("this_face_feature:", this_face_feature)
        print(feature_id)
        print(feature)
        feature_id.append('UnKnown')
        feature.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        print(feature_id)
        print(feature)
        min_diff = np.inf
        name = 'UnKnown'
        for i in range(len(feature_id)):
            dist = np.linalg.norm(np.array(feature[i]) - np.array(this_face_feature))
            print('dist: ', dist)
            if dist < min_diff:
                min_diff = dist
                name = feature_id[i]
        cv2.putText(frame, name, (face.left(), face.top() - 10), None, 1, (0, 255, 0), 2)
    else:
        pass
    return frame


def create_mysql():
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '123456',
        'database': 'face_recognition',
    }

    db = pymysql.connect(**config)
    cursor = db.cursor()

    sql_createTb = """CREATE TABLE face_feature2(
                      name VARCHAR(255),
                      face_width DOUBLE(7,3),
                      face_height DOUBLE(7,3),
                      mouse_width DOUBLE(7,3),
                      mouse_height DOUBLE(7,3),
                      nose_width DOUBLE(7,3),
                      nose_height DOUBLE(7,3),
                      left_eye_width DOUBLE(7,3),
                      left_eye_height DOUBLE(7,3),
                      right_eye_width DOUBLE(7,3),
                      right_eye_height DOUBLE(7,3))
                      """
    sql_createTb.encode('utf-8')
    print(sql_createTb)
    cursor.execute(sql_createTb)  # 只用创建一次，再次执行会出错


def write_to_mysql(name, feature):
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '123456',
        'database': 'face_recognition',
    }

    db = pymysql.connect(**config)
    cursor = db.cursor()

    sql_createTb = """INSERT INTO face_feature2 VALUES ('{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {})""".format(name,
                                                                                      feature[0],
                                                                                      feature[1],
                                                                                      feature[2],
                                                                                      feature[3],
                                                                                      feature[4],
                                                                                      feature[5],
                                                                                      feature[6],
                                                                                      feature[7],
                                                                                      feature[8],
                                                                                      feature[9])
    print(sql_createTb)
    try:
        cursor.execute(sql_createTb)
        db.commit()
    except Exception:
        db.rollback()
        print(Exception)
    db.close()

def read_from_mysql():
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '123456',
        'database': 'face_recognition',
    }

    db = pymysql.connect(**config)

    cursor = db.cursor()
    sql_createTb = 'SELECT * FROM face_feature2'
    all_data_num = cursor.execute(sql_createTb)
    all_data = cursor.fetchall()

    feature_id = []
    feature = []
    for i in range(len(all_data)):
        feature_id.append(all_data[i][0])
        feature.append(list(all_data[i][1:]))
    return feature_id, feature

if __name__ == '__main__':
    path = './data/2015年感官质量.xlsx'
    all_example_feature, chemistry_name, human_name, all_example_chemistry, all_example_human, all_example_score = get_data_1(path)
    all_example_chemistry = remove_nan(all_example_chemistry) #(91,111)
    all_example_human = remove_nan(all_example_human) #(91,8)  #all_example_score的shape是(91,)

    path = './data/卷烟检测结果.xlsx'
    all_example_feature2, chemistry_name2, all_example_chemistry2 = get_data_2(path)
    all_example_chemistry2 = remove_nan(all_example_chemistry2) #(43,103)

    #创建数据库 并且将数据写入数据库
    create_mysql(chemistry_name, human_name, chemistry_name2)
    feature_id = [item['example_name'] for item in all_example_feature]
    write_to_mysql(feature_id, all_example_chemistry , chemistry_name, 'chemistry_feature')
    write_to_mysql(feature_id, all_example_human, human_name, 'human_feature')
    feature_id2 = [item['example_name'] for item in all_example_feature2]
    write_to_mysql(feature_id2, all_example_chemistry2, chemistry_name2, 'chemistry_feature2')

    #重新从数据库中读取数据
    feature_id1, feature1 = read_from_mysql('chemistry_feature')
    feature_id2, feature2 = read_from_mysql('human_feature')
    feature_id3, feature3 = read_from_mysql('chemistry_feature2')

    #主成分分析
    new_all_example_human, pca_components_, pca_explained_variance_ratio_ = pca_analysis(all_example_human)

    #模糊综合评价法
    score = fuzzy_comprehensive_evaluation(all_example_chemistry, all_example_score, all_example_chemistry[0,:])
    print('score:', score)

    #神经网络
    x_train = copy.deepcopy(all_example_chemistry)
    y_train = copy.deepcopy(all_example_score)
    x_test = copy.deepcopy(all_example_chemistry[0,:])
    x_test = x_test[np.newaxis,:]
    predict = network(x_train, y_train, x_test)
    print('predict:', predict)