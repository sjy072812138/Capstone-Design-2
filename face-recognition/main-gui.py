# -*- coding: utf-8 -*-


import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import dlib
import cv2
from preprocess import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(626, 551)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 90, 201, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.clicked_button)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 200, 201, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.clicked_button_2)

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 300, 201, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.clicked_button_3)

        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(40, 370, 551, 91))
        self.tableView.setObjectName("tableView")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(140, 140, 104, 21))
        self.textEdit_3.setObjectName("textEdit_3")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(300, 71, 291, 281))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 10, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 490, 501, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 626, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        try:
            create_mysql()
        except:
            print('table already exists')

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "얼굴 입력"))
        self.pushButton_2.setText(_translate("MainWindow", "얼굴 인식"))
        self.pushButton_3.setText(_translate("MainWindow", "얼굴 데이터"))
        self.label.setText(_translate("MainWindow", "얼굴 인식 프로젝트"))
        self.label_2.setText(_translate("MainWindow",
                                        "Author:SunJiYao                        Contace:010-7638-6523                       "))

    def clicked_button(self):
        try:
            self.timer.stop()
        except:
            pass

        self.face_record_name = self.textEdit_3.toPlainText()
        self.face_record_img_path, ok = QFileDialog.getOpenFileName(self.centralwidget, "Choose img", r"C:",
                                                                    "*.png;;*.jpg;;All Files(*)")
        img = cv2.resize(cv2.imread(self.face_record_img_path), (280, 280))
        cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y, x = img.shape[:-1]
        frame = QImage(cvimg, x, y, QImage.Format_RGB888)
        self.scene = QGraphicsScene()
        self.scene.clear()
        self.pix = QPixmap.fromImage(frame)
        self.scene.addPixmap(self.pix)
        self.graphicsView.setScene(self.scene)
        frame, fv = face_recognition(pic_path=self.face_record_img_path)
        write_to_mysql(self.face_record_name, fv)

    def help(self):
        feature_id, feature = read_from_mysql()
        print(feature_id)
        print(feature)

        self.scene = QGraphicsScene()
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        img = face_recognition_name_from_frame(frame, feature_id, feature)
        img = cv2.resize(img, (280, 280))
        cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y, x = img.shape[:-1]
        frame = QImage(cvimg, x, y, QImage.Format_RGB888)
        self.scene.clear()
        self.pix = QPixmap.fromImage(frame)
        self.scene.addPixmap(self.pix)
        self.graphicsView.setScene(self.scene)
        cv2.destroyAllWindows()
        # if img_video == 'img':
        #     self.face_record_img_path, ok = QFileDialog.getOpenFileName(self.centralwidget, "选择图片",
        #                                                                 r"C:", "*.png;;*.jpg;;All Files(*)")
        #     img = face_recognition_name(pic_path=self.face_record_img_path, feature_id=feature_id, feature=feature)
        #     img = cv2.resize(img, (280, 280))
        #     cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     y, x = img.shape[:-1]
        #     frame = QImage(cvimg, x, y, QImage.Format_RGB888)
        #     self.scene = QGraphicsScene()
        #     self.scene.clear()
        #     self.pix = QPixmap.fromImage(frame)
        #     self.scene.addPixmap(self.pix)
        #     self.graphicsView.setScene(self.scene)
        # elif img_video == 'video':
        #     self.scene = QGraphicsScene()
        #     while True:
        #         capture = cv2.VideoCapture(0)
        #
        #         ret, frame = capture.read()
        #         img = face_recognition_name_from_frame(frame, feature_id, feature)
        #         print('now1:', img.shape)
        #         img = cv2.resize(img, (280, 280))
        #         print('now2:', img.shape)
        #         cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #         y, x = img.shape[:-1]
        #         frame = QImage(cvimg, x, y, QImage.Format_RGB888)
        #         self.scene.clear()
        #         self.pix = QPixmap.fromImage(frame)
        #         self.scene.addPixmap(self.pix)
        #         self.graphicsView.setScene(self.scene)
        #         print('setScene finish')
        #         time.sleep(2)
        #     cv2.destroyAllWindows()

    def clicked_button_2(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.help)
        self.timer.start(0.5)

    def clicked_button_3(self):
        try:
            self.timer.stop()
        except:
            pass

        feature_id, feature = read_from_mysql()
        self.model = QStandardItemModel(len(feature_id), 11)
        self.model.setHorizontalHeaderLabels(['name', 'face_width', 'face_height',
                                              'mouse_width', 'mouse_height',
                                              'nose_width', 'nose_height',
                                              'left_eye_width', 'left_eye_height',
                                              'right_eye_width', 'right_eye_height'])
        self.tableView.setModel(self.model)

        for i in range(len(feature_id)):
            item1 = QStandardItem(feature_id[i])
            item2 = QStandardItem(str(feature[i][0]))
            item3 = QStandardItem(str(feature[i][1]))
            item4 = QStandardItem(str(feature[i][2]))
            item5 = QStandardItem(str(feature[i][3]))
            item6 = QStandardItem(str(feature[i][4]))
            item7 = QStandardItem(str(feature[i][5]))
            item8 = QStandardItem(str(feature[i][6]))
            item9 = QStandardItem(str(feature[i][7]))
            item10 = QStandardItem(str(feature[i][8]))
            item11 = QStandardItem(str(feature[i][9]))
            self.model.setItem(i, 0, item1)
            self.model.setItem(i, 1, item2)
            self.model.setItem(i, 2, item3)
            self.model.setItem(i, 3, item4)
            self.model.setItem(i, 4, item5)
            self.model.setItem(i, 5, item6)
            self.model.setItem(i, 6, item7)
            self.model.setItem(i, 7, item8)
            self.model.setItem(i, 8, item9)
            self.model.setItem(i, 9, item10)
            self.model.setItem(i, 10, item11)


import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
