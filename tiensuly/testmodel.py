import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision import models
import os
from torchvision.models import ResNet50_Weights


class TransferLearningModel(nn.Module):
    def __init__(self, num_classes):
        super(TransferLearningModel, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Đóng băng các lớp pretrained
        for param in self.model.parameters():
            param.requires_grad = False

        # Thay thế Fully Connected Layer cuối
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class StudentRecognitionSystem:
    def __init__(self, model_path, data_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.data_path = data_path
        self.class_names = self.load_class_names()
        self.num_classes = len(self.class_names)
        if self.num_classes == 0:
            raise ValueError("No class directories found in the data path.")
        self.model = self.load_model().to(self.device)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_class_names(self):
        try:
            return [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        except Exception as e:
            print(f"Error loading class names: {e}")
            return []

    def load_model(self):
        try:
            model = TransferLearningModel(self.num_classes)
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

    def preprocess_face(self, face_image):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        return transform(face_image_pil).unsqueeze(0).to(self.device)

    def predict(self, face_image):
        image = self.preprocess_face(face_image)
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities

    def start_webcam_recognition(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to access webcam.")
            return

        print("Press 'q' to quit the webcam.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Unable to read from webcam.")
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

                if len(faces) == 0:
                    cv2.putText(frame, "No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                for (x, y, w, h) in faces:
                    face_image = frame[y:y + h, x:x + w]

                    if face_image.shape[0] < 60 or face_image.shape[1] < 60:
                        cv2.putText(frame, "Face too small", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        continue

                    probabilities = self.predict(face_image)
                    predicted_class = probabilities.argmax(dim=1).item()
                    similarity_score = probabilities[0][predicted_class].item() * 100

                    label = "Unknown"
                    if similarity_score >= 50:  # Ngưỡng nhận diện 50%
                        label = f'{self.class_names[predicted_class]}: {similarity_score:.2f}%'

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting webcam recognition.")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


# Đường dẫn đến mô hình và dữ liệu sinh viên
MODEL_PATH = r'C:\Users\admin\PycharmProjects\Diemdanh\tiensuly\checkpoints\best_model.pth'
DATA_PATH = r'C:\Users\admin\Desktop\video\anhsinhvien'

# Khởi tạo hệ thống và chạy nhận diện
recognition_system = StudentRecognitionSystem(MODEL_PATH, DATA_PATH)
recognition_system.start_webcam_recognition()
