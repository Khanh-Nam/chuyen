import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.models import ResNet50_Weights

# Tắt cảnh báo oneDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Tắt oneDNN optimizations

# Tắt cảnh báo từ TensorFlow
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Định nghĩa mô hình chuyển giao học tập
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes):
        super(TransferLearningModel, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False  # Đóng băng toàn bộ các lớp
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Hàm huấn luyện 1 epoch
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_labels, all_preds


# Hàm đánh giá 1 epoch
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_labels, all_preds


# Hàm chính để huấn luyện mô hình
def train_model_with_transfer(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.00002
    weight_decay = 1e-2
    min_delta = 0.01
    patience = 5  # Early stopping

    # Các phép biến đổi dữ liệu
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(size=128, scale=(0.5, 1.0)),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.3),
        transforms.GaussianBlur(5, sigma=(0.1, 2.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Chuẩn bị dữ liệu
    dataset = datasets.ImageFolder(data_path, transform=transform_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(dataset.classes)
    model = TransferLearningModel(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Huấn luyện và đánh giá
        train_loss, train_labels, train_preds = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_labels, val_preds = validate_epoch(model, val_loader, criterion, device)

        val_accuracy = sum([pred == label for pred, label in zip(val_preds, val_labels)]) / len(val_labels)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy * 100)

        precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy * 100:.2f}%, '
              f'Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, '
              f'F1 Score: {f1:.4f}')

        scheduler.step(val_loss)

        # Lưu mô hình tốt nhất
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print(f'Final Validation Accuracy: {val_accuracy * 100:.2f}%')

    # Vẽ đồ thị loss và accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.legend()
    plt.show()


# Gọi hàm huấn luyện
data_path = 'C:/Users/admin/Desktop/video/anhsinhvien'
train_model_with_transfer(data_path)
