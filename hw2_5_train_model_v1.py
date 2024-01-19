import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchsummary import summary
import matplotlib.pyplot as plt

# 定義數據轉換
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

# 載入驗證資料集
validation_dataset_with_erasing = torchvision.datasets.ImageFolder(root='./Dataset_OpenCvDl_Hw2_Q5/dataset/validation_dataset', transform=transform_with_erasing)
validation_loader_with_erasing = torch.utils.data.DataLoader(validation_dataset_with_erasing, batch_size=32, shuffle=False)

validation_dataset_without_erasing = torchvision.datasets.ImageFolder(root='./Dataset_OpenCvDl_Hw2_Q5/dataset/validation_dataset', transform=transform_without_erasing)
validation_loader_without_erasing = torch.utils.data.DataLoader(validation_dataset_without_erasing, batch_size=32, shuffle=False)

# 初始化模型
model_with_erasing = resnet50(pretrained=True)
model_without_erasing = resnet50(pretrained=True)

# 更換輸出層
num_ftrs = model_with_erasing.fc.in_features
model_with_erasing.fc = torch.nn.Linear(num_ftrs, 1)
model_without_erasing.fc = torch.nn.Linear(num_ftrs, 1)

# 打印模型結構
print("Model with Random Erasing:")
summary(model_with_erasing, (3, 224, 224))
print("\nModel without Random Erasing:")
summary(model_without_erasing, (3, 224, 224))

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

# 驗證模型
accuracy_with_erasing = validate_model(model_with_erasing, validation_loader_with_erasing)
accuracy_without_erasing = validate_model(model_without_erasing, validation_loader_without_erasing)

# 繪製條形圖
plt.bar(['With Erasing', 'Without Erasing'], [accuracy_with_erasing, accuracy_without_erasing])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy with and without Random Erasing')
plt.savefig('./resnet50_model_accuracy_comparison.png')

# 儲存模型權重
torch.save(model_with_erasing.state_dict(), './resnet50_model_with_erasing.pth')
torch.save(model_without_erasing.state_dict(), './resnet50_model_without_erasing.pth')

# 返回儲存的檔案路徑
'/mnt/data/model_accuracy_comparison.png', '/mnt/data/model_with_erasing.pth', '/mnt/data/model_without_erasing.pth'
