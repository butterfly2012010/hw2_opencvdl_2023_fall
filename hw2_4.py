import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPixmap, QImage, QPen

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm



class GraffitiBoard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        # 加載模型
        # 加載並修改 VGG19 模型
        self.model = models.vgg19_bn(pretrained=False)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1) # 更改為單通道輸入
        self.model.classifier[6] = nn.Linear(4096, 10)  # 修改最後一層為 10 個輸出
        self.model.load_state_dict(torch.load('best_vgg19_bn_mnist.pth'))
        self.model.eval()

    def initUI(self):
        self.setGeometry(100, 100, 280, 280)
        self.setWindowTitle('Draw a number')

        # 繪圖板
        self.canvas = QPixmap(280, 280)
        self.canvas.fill(QtCore.Qt.black)
        self.label = QLabel(self)
        self.label.setPixmap(self.canvas)

        # 預測按鈕
        self.predict_btn = QPushButton('Predict', self)
        self.predict_btn.clicked.connect(self.predict)

        # 重置按鈕
        self.reset_btn = QPushButton('Reset', self)
        self.reset_btn.clicked.connect(self.reset)

        # 用於顯示預測結果的標籤
        self.result_label = QLabel('Prediction: None', self)

        self.show()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.label.pixmap())
            p = QPen()
            p.setWidth(10)
            p.setColor(QtGui.QColor('white'))
            painter.setPen(p)
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = False

    def predict(self):
        # 將 QPixmap 對象轉換為 QImage
        image = self.label.pixmap().toImage()
        image = image.convertToFormat(QImage.Format_Grayscale8)

        # QImage 轉換為 PIL Image，並進行必要的轉換
        image = image.constBits().asarray()
        image = transforms.ToPILImage()(image)
        image = transforms.Resize((28, 28))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.5,), (0.5,))(image)
        image = image.unsqueeze(0)  # 增加批次維度

        # 進行推理
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy().flatten()

        # 更新 GUI 顯示預測結果
        self.result_label.setText(f'Prediction: {predicted}')

        # 顯示概率分布直方圖
        plt.figure()
        plt.bar(range(10), probabilities)
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.title('Probability Distribution')
        plt.show()

    def reset(self):
        # 清除繪圖板
        self.canvas.fill(QtCore.Qt.black)
        self.label.setPixmap(self.canvas)
        self.result_label.setText('Prediction: None')

    def resizeImage(self, image, size):
        # PIL Image 轉換為 Torch Tensor
        image = transforms.ToPILImage()(image)
        image = transforms.Resize(size)(image)
        image = transforms.ToTensor()(image)
        return image.unsqueeze(0)  # 增加批次維度

# PyQt 应用程序的主函数
def main():
    app = QApplication(sys.argv)
    ex = GraffitiBoard()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

