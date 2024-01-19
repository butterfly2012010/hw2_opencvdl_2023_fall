import argparse
import parser

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm

# 參數解析器
parser = argparse.ArgumentParser()
# 要不要訓練模型
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

# 假設資料集路徑
dataset_path = "./Dataset_OpenCvDl_Hw2_Q5/dataset"

# 設定兩種數據轉換，一個包含 RandomErasing，另一個不包含
transform_with_erasing = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(),
])

transform_without_erasing = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

# 加載訓練資料集
train_dataset_with_erasing = torchvision.datasets.ImageFolder(
    root=f"{dataset_path}/training_dataset", 
    transform=transform_with_erasing
)

train_dataset_without_erasing = torchvision.datasets.ImageFolder(
    root=f"{dataset_path}/training_dataset", 
    transform=transform_without_erasing
)

# 加載驗證資料集
validation_dataset_with_erasing = torchvision.datasets.ImageFolder(
    root=f"{dataset_path}/validation_dataset", 
    transform=transform_with_erasing
)

validation_dataset_without_erasing = torchvision.datasets.ImageFolder(
    root=f"{dataset_path}/validation_dataset",
    transform=transform_without_erasing
)

# 創建數據加載器
train_loader_with_erasing = DataLoader(train_dataset_with_erasing, batch_size=32, shuffle=True)
train_loader_without_erasing = DataLoader(train_dataset_without_erasing, batch_size=32, shuffle=True)
validation_loader_with_erasing = DataLoader(validation_dataset_with_erasing, batch_size=32, shuffle=False)
validation_loader_without_erasing = DataLoader(validation_dataset_without_erasing, batch_size=32, shuffle=False)

# 初始化模型
model_with_erasing = resnet50(pretrained=False)
model_without_erasing = resnet50(pretrained=False)

# 修改模型的輸出層
num_ftrs = model_with_erasing.fc.in_features
model_with_erasing.fc = nn.Sequential(nn.Linear(num_ftrs, 1, bias=True), nn.Sigmoid())
model_without_erasing.fc = nn.Sequential(nn.Linear(num_ftrs, 1, bias=True), nn.Sigmoid())

# 打印模型結構
print("Model with Random Erasing:")
summary(model_with_erasing, (3, 224, 224), device="cpu")
print("\nModel without Random Erasing:")
summary(model_without_erasing, (3, 224, 224), device="cpu")

# train_and_validate_model 函數
def train_and_validate_model(model, train_loader, val_loader, epochs=5, learning_rate=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 用於存儲每個 epoch 的訓練和驗證損失及準確率
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(epochs)):
        # 訓練階段
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.type_as(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels.unsqueeze(1)).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100 * train_correct / total_train

        # 驗證階段
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.type_as(inputs)

                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels.unsqueeze(1)).sum().item()
                total_val += labels.size(0)

        val_accuracy = 100 * val_correct / total_val

        # 計算平均損失和準確率
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

    return model#, train_losses, val_losses, train_accuracies, val_accuracies


# 訓練模型
if args.train:
    trained_model_with_erasing = train_and_validate_model(model_with_erasing, train_loader_with_erasing, validation_loader_with_erasing, epochs=10, learning_rate=0.001)
    trained_model_without_erasing = train_and_validate_model(model_without_erasing, train_loader_without_erasing, validation_loader_without_erasing, epochs=10, learning_rate=0.001)

    # 儲存模型權重 (在模型訓練完成後進行)
    torch.save(trained_model_with_erasing.state_dict(), "resnet50_model_with_erasing.pth")
    torch.save(trained_model_without_erasing.state_dict(), "resnet50_model_without_erasing.pth")
else:
    trained_model_with_erasing = resnet50(pretrained=False)
    trained_model_without_erasing = resnet50(pretrained=False)
    # 修改模型的輸出層
    num_ftrs = trained_model_with_erasing.fc.in_features
    trained_model_with_erasing.fc = nn.Sequential(nn.Linear(num_ftrs, 1, bias=True), nn.Sigmoid())
    trained_model_without_erasing.fc = nn.Sequential(nn.Linear(num_ftrs, 1, bias=True), nn.Sigmoid())
    # 載入模型權重
    trained_model_with_erasing.load_state_dict(torch.load("resnet50_model_with_erasing.pth"))
    trained_model_without_erasing.load_state_dict(torch.load("resnet50_model_without_erasing.pth"))

# 驗證函數
def validate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 驗證模型和繪製條形圖的部分將在訓練函數完成後執行
accuracy_with_erasing = validate_model(trained_model_with_erasing, validation_loader_with_erasing)
accuracy_without_erasing = validate_model(trained_model_without_erasing, validation_loader_without_erasing)

# 繪製條形圖
plt.bar(["With Erasing", "Without Erasing"], [accuracy_with_erasing, accuracy_without_erasing])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.savefig("resnet50_model_accuracy_comparison.png")

