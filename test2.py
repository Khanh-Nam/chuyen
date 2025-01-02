import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
import pickle
from tensorflow.keras.models import load_model
import time
import os


# Hàm trích xuất embedding từ ảnh
def get_embedding(img_path):
    try:
        embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
        return np.array(embedding[0]["embedding"])
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


# Hàm dự đoán từ ảnh
def predict(image, clf_model, label_encoder):
    embedding = get_embedding(image)
    if embedding is None:
        return None, None  # Nếu không lấy được embedding

    embedding = embedding.reshape(1, -1)  # Đảm bảo rằng embedding có dạng đúng cho mô hình

    prediction = clf_model.predict(embedding)
    predicted_class_index = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

    probability = prediction[0][predicted_class_index]

    return predicted_class, probability


# Đường dẫn đến mô hình đã lưu
model_path = r"C:\Users\admin\PycharmProjects\Diemdanh\tiensuly\checkpoints\classification_model.h5"
label_encoder_path = r"C:\Users\admin\PycharmProjects\Diemdanh\tiensuly\checkpoints\label_encoder.pkl"

# Tải mô hình và label encoder
clf_model = load_model(model_path)
with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

# Khởi tạo OpenCV để mở camera
cap = cv2.VideoCapture(0)

# Dựng bộ phát hiện khuôn mặt từ OpenCV (Haar Cascade hoặc DNN)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_counter = 0  # Biến đếm số lượng frame đã xử lý
prev_time = time.time()  # Để tính FPS

# Tạo thư mục lưu ảnh nếu chưa có
output_dir = 'captured_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Giảm độ phân giải để xử lý nhanh hơn
    frame = cv2.resize(frame, (640, 480))

    # Chuyển đổi ảnh thành màu xám (grayscale) để phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Duyệt qua tất cả các khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        # Vẽ khung vuông quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Cắt phần khuôn mặt từ ảnh
        face_img = frame[y:y + h, x:x + w]

        # Kiểm tra nếu frame_counter chia hết cho 5 (chỉ dự đoán mỗi 5 frame)
        if frame_counter % 5 == 0:
            predicted_class, probability = predict(face_img, clf_model, label_encoder)

            if predicted_class is not None and probability > 0.6:  # Kiểm tra tỷ lệ nhận diện > 60%
                # Hiển thị tên người nhận diện và xác suất trên khung
                probability_percent = probability * 100  # Chuyển đổi xác suất thành phần trăm
                cv2.putText(frame, f"{predicted_class}: {probability_percent:.2f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Lưu ảnh khi đạt ngưỡng nhận diện
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_name = f"{predicted_class}_{timestamp}.jpg"
                img_path = os.path.join(output_dir, img_name)
                cv2.imwrite(img_path, face_img)
                print(f"Captured: {img_path}")

    # Tính toán FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Hiển thị FPS trên video
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị video với các khuôn mặt nhận diện
    cv2.imshow('Face Recognition', frame)

    frame_counter += 1  # Tăng biến đếm mỗi vòng lặp

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
