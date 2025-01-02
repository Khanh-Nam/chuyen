import cv2
import os
import time
import numpy as np

# Hàm để kiểm tra góc khuôn mặt
def is_face_profile(face_image):
    # Có thể sử dụng một mô hình hoặc thuật toán để kiểm tra góc khuôn mặt
    # Ở đây ta chỉ làm giả định, trả về True nếu face_image không phải là góc quay lớn
    return True  # Thay thế bằng kiểm tra thực tế nếu cần

# Hàm chính để tách khuôn mặt từ video và resize về 512x512
def extract_faces_from_videos(base_folder):
    # Tạo thư mục để lưu ảnh
    output_base_dir = os.path.join(base_folder, 'anhsinhvien')
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Tải mô hình Haar Cascade cho khuôn mặt và mắt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Duyệt qua từng thư mục trong thư mục cơ sở
    for student_folder in os.listdir(base_folder):
        student_path = os.path.join(base_folder, student_folder)

        # Kiểm tra xem đó có phải là thư mục không
        if os.path.isdir(student_path):
            print(f"\nĐang xử lý thư mục: {student_folder}")

            # Tạo thư mục cho sinh viên nếu chưa tồn tại
            student_output_dir = os.path.join(output_base_dir, student_folder)
            if not os.path.exists(student_output_dir):
                os.makedirs(student_output_dir)

            # Duyệt qua tất cả các video trong thư mục sinh viên
            for video_file in os.listdir(student_path):
                video_path = os.path.join(student_path, video_file)

                # Kiểm tra xem file có phải là video không
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    print(f"Đang xử lý video: {video_file}")

                    cap = cv2.VideoCapture(video_path)

                    if not cap.isOpened():
                        print(f"Không thể mở video: {video_path}")
                        continue

                    saved_count = 0
                    total_faces_detected = 0  # Số lượng khuôn mặt phát hiện

                    while True:
                        ret, frame = cap.read()

                        if not ret:
                            print("Không còn frame nào để đọc. Dừng lại.")
                            break  # Dừng nếu không còn frame nào

                        # Tiền xử lý: chuyển đổi sang ảnh xám và tăng cường độ tương phản
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray_frame = cv2.equalizeHist(gray_frame)

                        # Phát hiện khuôn mặt
                        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=7)

                        # Nếu phát hiện khuôn mặt
                        if len(faces) > 0:
                            total_faces_detected += len(faces)  # Tăng số lượng khuôn mặt phát hiện
                            for (x, y, w, h) in faces:
                                # Kiểm tra tỷ lệ khuôn mặt (gần với hình chữ nhật)
                                if w > 30 and h > 30 and (0.8 < w / h < 1.2):
                                    # Kiểm tra có hai mắt trong khuôn mặt
                                    face_region = gray_frame[y:y + h, x:x + w]
                                    eyes = eye_cascade.detectMultiScale(face_region)

                                    if len(eyes) >= 2:  # Nếu phát hiện ít nhất hai mắt
                                        face_image = frame[y:y + h, x:x + w]

                                        # Kiểm tra góc khuôn mặt
                                        if not is_face_profile(face_image):
                                            continue  # Bỏ qua nếu góc quá lớn

                                        # Resize khuôn mặt về kích thước 512x512
                                        face_image_resized = cv2.resize(face_image, (512, 512))

                                        # Tăng cường ánh sáng
                                        face_image_resized = cv2.convertScaleAbs(face_image_resized, alpha=1.2, beta=30)

                                        # Lưu ảnh đã resize nhiều lần, thêm timestamp để tránh ghi đè
                                        for i in range(3):  # Giả sử bạn muốn lưu mỗi ảnh 3 lần
                                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                                            face_filename = os.path.join(student_output_dir,
                                                                         f'{student_folder}_face_{saved_count}_{timestamp}_{i}.jpg')
                                            # Lưu ảnh đã resize
                                            cv2.imwrite(face_filename, face_image_resized)
                                            saved_count += 1  # Đếm số lượng ảnh đã lưu cho mỗi video

                                        # Vẽ hình chữ nhật quanh khuôn mặt (tuỳ chọn)
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Hiển thị khung hình video với khuôn mặt đã được phát hiện
                        cv2.imshow("Video", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break  # Nhấn 'q' để thoát

                    cap.release()
                    cv2.destroyAllWindows()
                    accuracy_rate = (saved_count / total_faces_detected * 100) if total_faces_detected > 0 else 0
                    accuracy_rate = min(accuracy_rate, 100)  # Đảm bảo tỷ lệ chính xác không vượt quá 100%
                    print(
                        f'Đã lưu {saved_count} ảnh khuôn mặt từ video "{video_file}" vào thư mục "{student_output_dir}".')
                    print(f'Tỷ lệ phát hiện chính xác: {accuracy_rate:.2f}% ({saved_count}/{total_faces_detected})')

if __name__ == "__main__":
    # Bắt đầu thời gian
    start_time = time.time()

    base_folder = input("Nhập đường dẫn đến thư mục chứa các thư mục sinh viên: ")
    base_folder = base_folder.strip()  # Loại bỏ khoảng trắng đầu và cuối

    extract_faces_from_videos(base_folder)

    # Kết thúc thời gian
    print(f'Thời gian thực thi: {time.time() - start_time:.2f} giây')
