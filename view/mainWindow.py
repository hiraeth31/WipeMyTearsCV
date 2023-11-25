import os
import torch
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog
import sys
from datetime import datetime

model = torch.hub.load("ultralytics/yolov5", "yolov5s")
# app = QtWidgets.QApplication(sys.argv)
# window = QtWidgets.QWidget()
# window.setWindowTitle("WipeMyTearsCV")
# window.resize(1200, 700)
# label = QtWidgets.QLabel("<center>Привет, мир!</center>")
# btnQuit = QtWidgets.QPushButton("Закрыть окно")
# vbox = QtWidgets.QVBoxLayout()
# vbox.addWidget(label)
# vbox.addWidget(btnQuit)
# window.setLayout(vbox)
# btnQuit.clicked.connect(app.quit)
# window.show()
# sys.exit(app.exec())

def chooseFolder():
    path = QFileDialog.getExistingDirectory(window, "Выберите папку для проверки")
    if path:
        detectCars(path)

def detectCars(folderPath):
    files = os.listdir(folderPath)
    currentDate = datetime.now()
    timestamp = currentDate.strftime("%Y-%m-%d_%H%M%S")
    outputFolder = "../res"
    newFolderPath = os.path.join(outputFolder, timestamp)
    for file in files:
        fullPath = os.path.join(folderPath, file)
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            result = model(fullPath)
            result.save(save_dir=folderPath + "/result")
            # data = [fullPath]
            # window.tableViewResults.appendRow(data)
            result.show()

def viewResultsFolder():
    currentDate = datetime.now()
    timestamp = currentDate.strftime("%Y-%m-%d_%H%M%S")
    outputFolder = "Output"
    newFolderPath = os.path.join(outputFolder, timestamp)

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("MainWindow.ui")

window.btnExit.clicked.connect(app.quit)
window.btnDetect.clicked.connect(chooseFolder)


window.show()
sys.exit(app.exec())


