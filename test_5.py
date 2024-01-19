import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = qtg.QImage(image_path)
        image = image.scaled(224, 224, qtc.Qt.KeepAspectRatio)
        image = transforms.ToTensor()(image)
        return image

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OpenCvDl Hw2 Q5")
        self.setFixedSize(800, 600)

        self.inference_dataset = InferenceDataset("./Dataset_OpenCvDl_Hw2_Q5/dataset/inference_dataset")

        self.show_images_button = qtw.QPushButton("1. Show Images")
        self.show_images_button.clicked.connect(self.show_images)

        self.central_widget = qtw.QWidget()
        self.layout = qtw.QVBoxLayout()
        self.layout.addWidget(self.show_images_button)
        self.central_widget.setLayout(self.layout)

        self.setCentralWidget(self.central_widget)

    def show_images(self):
        # Get 1 image from each class in the inference dataset
        cat_image, dog_image = self.inference_dataset[0], self.inference_dataset[1]

        # Show images in a new window
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cat_image.permute(1, 2, 0))
        axes[0].set_title("Cat")
        axes[1].imshow(dog_image.permute(1, 2, 0))
        axes[1].set_title("Dog")
        plt.show()

if __name__ == "__main__":
    app = qtw.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()