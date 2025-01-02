import os
from PIL import Image

# Đường dẫn tới thư mục ảnh sinh viên
image_dir = r'C:\Users\admin\Desktop\video\anhsinhvien'  # Thay thế bằng đường dẫn thực tế của bạn

def check_image_quality(image_dir):
    resolutions = set()  # Lưu các độ phân giải khác nhau
    formats = set()      # Lưu các định dạng màu khác nhau
    all_images_valid = True

    # Duyệt qua tất cả ảnh trong thư mục và các thư mục con
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):  # Chỉ xét định dạng ảnh
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        resolutions.add(img.size)  # Lưu độ phân giải của ảnh
                        formats.add(img.mode)      # Lưu định dạng màu của ảnh

                        # Kiểm tra độ phân giải tối thiểu là 128x128
                        if img.size[0] < 128 or img.size[1] < 128:
                            print(f"Ảnh {file_path} có độ phân giải thấp: {img.size}")
                            all_images_valid = False
                except Exception as e:
                    print(f"Không thể mở ảnh {file_path}: {e}")
                    all_images_valid = False

    # Kết quả kiểm tra
    print("Độ phân giải duy nhất trong tập dữ liệu:", resolutions)
    print("Các định dạng màu của ảnh:", formats)
    if all_images_valid:
        print("Tất cả ảnh đều phù hợp để huấn luyện!")
    else:
        print("Một số ảnh có thể không phù hợp để huấn luyện.")

# Gọi hàm kiểm tra
check_image_quality(image_dir)
