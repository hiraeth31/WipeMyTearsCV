import csv
import sys
import numpy as np
import torch
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTableView, QPushButton, QHBoxLayout, QFileDialog, \
    QLabel
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PIL import Image
import os
import cv2
from sklearn.cluster import KMeans


def extract_colors(image_path, num_colors=1):
    image = cv2.imread(image_path)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    pixels = image_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    results.show()
    return dominant_colors


class MyApplication(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def find_car(self, input_dir, output_cars='output.csv'):
        cars, imgs = ['car', 'truck', 'bus'], []
        for file_name in os.listdir(input_dir):
            imgs.append(cv2.imread(os.path.join(input_dir, file_name)))
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        results = model(imgs)
        output_folder = 'Datasets/images'
        os.makedirs(output_folder, exist_ok=True)
        with open(output_cars, 'w', newline='') as f:
            writer = csv.writer(f)
            for i, file_name in enumerate(os.listdir(input_dir)):
                res = [n in results.pandas().xyxy[i]['name'].unique() for n in cars]
                has_car = bool(sum(res))
                writer.writerow([file_name, has_car])
                if has_car:
                    output_path = os.path.join(output_folder, file_name)
                    cv2.imwrite(output_path, imgs[i])
                    print(f"Фото с машиной сохранено: {output_path}")
                    image_path = os.path.join(input_dir, file_name)
                    image = Image.open(image_path)
                    width, height = image.size
                    size_in_bytes = os.path.getsize(image_path)
                    size_in_mbytes = float(size_in_bytes / 1048576)
                    self.update_table(file_name, width, height, size_in_mbytes)

    def findcar_onimage(self):
        self.find_car('datatest')

    def detectButtonClicked(self):
        selected_index = self.table_view.selectionModel().currentIndex()
        if selected_index.isValid():
            selected_data = selected_index.siblingAtColumn(0).data()
            image_path = rf'C:\Users\Nosferatu\PycharmProjects\WipeMyTearsCV\Datasets\images\{selected_data}'
            dominant_colors = extract_colors(image_path)
            print("Dominant Colors:")
            for color in dominant_colors:
                print(f"RGB: {color}")
            message = f"{selected_data}\n" + str(dominant_colors)
            self.result_label.setText(f"Выбранная запись: {message}")
        else:
            self.result_label.setText("Выберите строку в таблице")

    def init_ui(self):
        self.model = QStandardItemModel(0, 3)
        self.model.setHorizontalHeaderLabels(["Название", "Разрешение", "Вес"])
        self.result_label = QLabel("Выбранная запись:")
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        btn_view_result = QPushButton('Выбрать директорию', self)
        btn_detection = QPushButton('Просмотреть', self)
        btn_exit = QPushButton('Выход', self)
        detect_button = QPushButton("Просмотреть)")
        find_button = QPushButton("Find Car")
        btn_view_result.clicked.connect(self.view_result)
        btn_detection.clicked.connect(self.detect)
        btn_exit.clicked.connect(self.exit_app)
        detect_button.clicked.connect(self.detectButtonClicked)
        find_button.clicked.connect(self.findcar_onimage)
        button_layout = QHBoxLayout()
        button_layout.addWidget(btn_view_result)
        button_layout.addWidget(btn_detection)
        button_layout.addWidget(detect_button)
        button_layout.addWidget(btn_exit)
        # button_layout.addWidget(find_button)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table_view)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_label)
        self.setLayout(main_layout)
        self.setFixedSize(1200, 700)
        self.setWindowTitle('WipeMyTearsCV')
        self.show()

    def view_result(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Выберите папку с изображениями')
        if folder_path:
            self.find_car(folder_path)

    def update_table(self, file_name, width, height, size_in_mbytes):
        # Добавляем данные в таблицу
        row_position = self.model.rowCount()
        self.model.insertRow(row_position)
        self.model.setItem(row_position, 0, QStandardItem(file_name))
        self.model.setItem(row_position, 1, QStandardItem(f"{width}x{height}"))
        self.model.setItem(row_position, 2, QStandardItem(f"{round(size_in_mbytes, 1)} Mbytes"))

    def detect(self):
        selected_index = self.table_view.selectionModel().currentIndex()
        if selected_index.isValid():
            selected_data = selected_index.siblingAtColumn(0).data()
            image_path = rf'C:\Users\Nosferatu\PycharmProjects\WipeMyTearsCV\Datasets\images\{selected_data}'
            img = cv2.imread(image_path)
            color_bar, color = self.calc_metric(img, 100, 600, 900, 250)
            cv2.imshow('Цвет', color_bar)
            cv2.waitKey(0)

    def create_bar(self, height, width, color):
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])

        return bar, (red, green, blue)

    def calc_metric(self, image, x, y, w, h):
        # image = image[(y - h):y, x:(x + w)]
        cv2.imshow('Изображение', image)
        data = np.float32(image).reshape((-1, 3))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        percentages = (np.unique(label, return_counts=True)[1]) / data.shape[0]

        bar, color = self.create_bar(400, 400, center[np.argmax(percentages)])
        img_bar = np.hstack((bar,))

        return img_bar, color

    def exit_app(self):
        sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApplication()
    sys.exit(app.exec())
