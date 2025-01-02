import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from torchvision.models import resnet18, ResNet18_Weights

# Sử dụng weights thay vì pretrained=True
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

def train_model_with_pretrained(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.0001
    weight_decay = 1e-4
    min_delta = 0.02
    patience = 10

    # Dữ liệu augmentation và preprocess
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomErasing(p=0.5),  # Thêm Random Erasing
        transforms.ToTensor(),  # Chuyển đổi ảnh thành tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        dataset = datasets.ImageFolder(data_path, transform=transform_train)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        num_classes = len(dataset.classes)

        # Tải mô hình ResNet18 tiền huấn luyện và thay đổi lớp cuối
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 64),  # Giảm số nơ-ron
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Giảm dropout để kiểm tra
            nn.Linear(64, num_classes)
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        train_losses = []
        val_losses = []
        val_accuracies = []

        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            model.train()
            epoch_loss = 0
            all_labels = []
            all_preds = []
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                optimizer.step()

                all_labels.extend(labels.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())

            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            val_labels = []
            val_preds = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    val_accuracy += (predicted == labels).sum().item()

                    val_labels.extend(labels.cpu().numpy())
                    val_preds.extend(predicted.cpu().numpy())

            val_loss /= len(val_loader)
            val_accuracy /= len(val_dataset)

            train_losses.append(epoch_loss / len(train_loader))
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy * 100)

            precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
            recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
            f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Loss: {epoch_loss / len(train_loader):.4f}, '
                  f'Validation Loss: {val_loss:.4f}, '
                  f'Validation Accuracy: {val_accuracy * 100:.2f}%, '
                  f'Precision: {precision:.4f}, '
                  f'Recall: {recall:.4f}, '
                  f'F1 Score: {f1:.4f}')

            scheduler.step(val_loss)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered due to lack of improvement.")
                break

        print(f'Final Validation Accuracy: {val_accuracy * 100:.2f}%')

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

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
data_path = 'C:/Users/admin/Desktop/video/anhsinhvien'
train_model_with_pretrained(data_path)
