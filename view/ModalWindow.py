import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, \
    QLabel, QDialog
from PyQt6.QtGui import QPixmap, QColor
import cv2
import numpy as np


class ImageInfoDialog(QDialog):
    def __init__(self, image_path, dominant_colors, file_name, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Информация о картинке")
        layout = QVBoxLayout()
        # self.layout.setStyleSheet("QVBoxLayout { border: 2px solid #C53D46; }")
        image_label = QLabel()

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        new_path = '../view/AiTest/'
        results = model(image_rgb)
        output_image = results.render()[0]
        # results.show()
        cv2.imwrite(new_path + file_name, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        img_path = new_path + file_name
        pixmap = QPixmap(img_path)
        image_label.setPixmap(pixmap.scaled(800, 500, Qt.AspectRatioMode.KeepAspectRatio))
        layout.addWidget(image_label)
        color_info_label = QLabel(f"Доминирующий цвет: {dominant_colors}")
        for i in dominant_colors:
            color = i
            self.color_square_label = QLabel()
            layout.addWidget(self.color_square_label)
            color = [min(max(c, 0), 255) for c in color]
            color_string = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            print(color_string)
            pixmap = QPixmap(100, 100)
            pixmap.fill(QColor(color_string))
            self.color_square_label.setPixmap(pixmap)
        layout.addWidget(color_info_label)
        self.setLayout(layout)