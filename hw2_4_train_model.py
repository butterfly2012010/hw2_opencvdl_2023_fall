import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm

# 數據加載和預處理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.RandomHorizontalFlip(p=0.5),  # p=0.5 means that the transformation has a 50% probability of being applied
    # transforms.RandomVerticalFlip(p=0.5),  # p=0.5 means that the transformation has a 50% probability of being applied
    # transforms.RandomRotation(30),  # Rotates the image by a maximum of 30 degrees
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載並修改 VGG19 模型
model = models.vgg19_bn(pretrained=False)
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1) # 更改為單通道輸入
model.classifier[6] = nn.Linear(4096, 10)  # 修改最後一層為 10 個輸出
model.to(device)

# 顯示模型結構
summary(model, (1, 32, 32))


# 設置損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 訓練和驗證函數
def train_and_validate(model, trainloader, testloader, epochs):
    best_acc = 0.0
    best_model_state = None
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss.append(running_loss / len(trainloader))
        train_acc.append(100 * correct / total)

        # 驗證階段
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss.append(running_loss / len(testloader))
        test_acc.append(100 * correct / total)

        # 檢查是否為最佳模型
        if test_acc[-1] > best_acc:
            best_acc = test_acc[-1]
            best_model_state = model.state_dict()

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.2f}%, Test Acc: {test_acc[-1]:.2f}%')

    # return train_loss, train_acc, test_loss, test_acc
    # 返回訓練和驗證結果以及最佳模型狀態
    return train_loss, train_acc, test_loss, test_acc, best_model_state

# 執行訓練和驗證，並獲取最佳模型
epochs = 30
train_loss, train_acc, test_loss, test_acc, best_model_state = train_and_validate(model, trainloader, testloader, epochs)

# 保存最佳模型狀態
torch.save(best_model_state, 'best_vgg19_bn_mnist.pth')

# 使用 matplotlib 繪製結果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.title('Loss over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

plt.savefig('vgg19_bn_mnist.png')
plt.show()