import os
import torch
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog
import sys
from datetime import datetime

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

def chooseFolder():
    path = QFileDialog.getExistingDirectory(window, "Выберите папку для проверки")
    if path:
        detectCars(path)


def detectCars(folderPath):
    files = os.listdir(folderPath)
    for file in files:
        fullPath = os.path.join(folderPath, file)
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            result = model(fullPath)
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


