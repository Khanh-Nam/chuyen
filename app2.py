from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from datetime import datetime, timedelta
from flask_mail import Mail, Message
import random
import string
import pandas as pd
import torch.nn as nn
from datetime import datetime

from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import os

import pickle
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, db, firestore

from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key for session management
import firebase_admin
from firebase_admin import credentials, db, firestore

# Khởi tạo Firebase Admin SDK
SERVICE_ACCOUNT_PATH = r'C:\\Users\\admin\\Downloads\\student-identification-s-dd4a5-firebase-adminsdk-u2pbn-57adc39235.json'

# URL của Realtime Database
REALTIME_DB_URL = 'https://student-identification-s-dd4a5-default-rtdb.firebaseio.com'

def initialize_firebase():
    """
    Khởi tạo Firebase, bao gồm Firestore và Realtime Database.
    """
    try:
        # Khởi tạo Firebase App nếu chưa được khởi tạo
        if not firebase_admin._apps:
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred, {
                'databaseURL': REALTIME_DB_URL
            })
        print("Firebase đã được khởi tạo thành công.")
    except Exception as e:
        print(f"Lỗi khi khởi tạo Firebase: {e}")

def get_firestore_client():
    """
    Kết nối và trả về client Firestore.
    """
    try:
        return firestore.client()
    except Exception as e:
        print(f"Lỗi khi kết nối Firestore: {e}")
        return None

def get_realtime_db_ref(ref_path: str):
    """
    Trả về tham chiếu (reference) đến một nhánh trong Realtime Database.
    :param ref_path: Đường dẫn tới nhánh trong Realtime Database.
    """
    try:
        return db.reference(ref_path)
    except Exception as e:
        print(f"Lỗi khi kết nối Realtime Database: {e}")
        return None
# Khởi tạo Firebase
initialize_firebase()

# Tạo các kết nối
firestore_db = get_firestore_client()  # Firestore client
realtime_ref = get_realtime_db_ref("attendance")  # Realtime Database (attendance branch)
class_ref = get_realtime_db_ref("class_information")  # Realtime Database (class_information branch)
# Kết nối Firestore
def initialize_firestore():
    from firebase_admin import credentials, initialize_app
    cred = credentials.Certificate(r'C:\\Users\\admin\\Downloads\\student-identification-s-dd4a5-firebase-adminsdk-u2pbn-57adc39235.json')
    if not firebase_admin._apps:
        initialize_app(cred)
    return firestore.client()

firestore_db = initialize_firestore()
@app.route('/class-info', methods=['GET'])
def class_info_page():
    """
    Router truyền dữ liệu đến HTML và in dữ liệu ra console
    """
    print("Data truyền đến HTML:", class_ref)  # In dữ liệu ra console để kiểm tra
    return render_template('class_info.html', class_info=class_information)


USER_COLLECTION = "users"

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # Thay bằng biến môi trường
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # Thay bằng biến môi trường
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

# Dummy database for demonstration purposes
users_db = {
    'testuser@example.com': {'password': 'testpassword', 'is_verified': True}
}

otp_storage = {}
# Mô hình Transfer Learning

class DeepFaceClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepFaceClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Tải mô hình DeepFace embedding và mô hình phân loại
deepface_model_path = r'C:\Users\admin\PycharmProjects\Diemdanh\tiensuly\checkpoints\classification_model.h5'
label_encoder_path = r'C:\Users\admin\PycharmProjects\Diemdanh\tiensuly\checkpoints\label_encoder.pkl'

# Load mô hình từ DeepFace
# from tensorflow.keras.models import load_model
classification_model = load_model(deepface_model_path)

# Load bộ mã hóa nhãn
with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

# Hàm trích xuất embedding từ ảnh bằng DeepFace
def get_embedding_deepface(image):
    from deepface import DeepFace
    embedding = DeepFace.represent(image, model_name="Facenet", enforce_detection=False)
    if embedding:
        return np.array(embedding[0]['embedding'])
    return None

# Load dữ liệu sinh viên
def load_student_data(directory):
    student_data = {}
    files = os.listdir(directory)
    for filename in files:
        student_info = filename.split('_')
        if len(student_info) >= 3:
            student_name = student_info[1]
            student_id = student_info[2]
            student_data[filename] = (student_name, student_id)
        else:
            print(f"Filename {filename} is invalid, missing information.")
    return student_data

student_data = load_student_data(r'C:\Users\admin\Desktop\video\anhsinhvien')

# Dự đoán sinh viên từ ảnh đã cắt
def predict_student(face_img):
    embedding = get_embedding_deepface(face_img)
    if embedding is None:
        return "Unknown", "Unknown ID", 0, "N/A"

    embedding = embedding.reshape(1, -1)  # Đảm bảo embedding đúng dạng cho mô hình
    probabilities = classification_model.predict(embedding)
    predicted_class_index = np.argmax(probabilities)
    confidence_percentage = probabilities[0][predicted_class_index] * 100

    if confidence_percentage < 40 or predicted_class_index >= len(student_data):
        return "Unknown", "Unknown ID", confidence_percentage, "N/A"

    predicted_student = list(student_data.values())[predicted_class_index]
    student_name, student_id = predicted_student

    return student_name, student_id, confidence_percentage, "N/A"

# Hàm lưu dữ liệu vào Firebase
def save_to_firebase(student_name, student_id, capture_time, attendance_type, duration=None):
    ref = db.reference(f"attendance/{student_id}/{capture_time}")
    attendance_data = {
        'Name': student_name,
        'Student ID': student_id,
        'Capture Time': capture_time,
        'Attendance Type': attendance_type,
        'Duration': duration if duration else None
    }
    try:
        ref.set(attendance_data)
        print(f"Dữ liệu đã được lưu vào Firebase cho sinh viên {student_id} vào {capture_time}")
    except Exception as e:
        print(f"Lỗi khi lưu vào Firebase: {e}")

# API nhận ảnh và trả về kết quả
@app.route('/capture_image', methods=['POST'])
def capture_image():
    image_data = request.form['image']
    if image_data.startswith("data:image"):
        image_data = image_data.split(",")[1]

    try:
        image_data = base64.b64decode(image_data)
        image_np = np.frombuffer(image_data, dtype=np.uint8)

        if image_np.size == 0:
            return jsonify({'error': 'Decoded image is empty'})

        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({'error': 'No face detected!'})

        if len(faces) > 1:
            return jsonify({'error': 'Multiple faces detected! Only one person is allowed for attendance.'})

        # If exactly one face is detected
        x, y, w, h = faces[0]
        face_img = img[y:y + h, x:x + w]

        student_name, student_id, confidence_percentage, prediction_time = predict_student(face_img)

        if confidence_percentage >= 90:
            capture_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            attendance_type = request.form.get('status', 'Check-in')
            save_to_firebase(student_name, student_id, capture_time, attendance_type)
            response = {
                'image': f'data:image/jpeg;base64,{base64.b64encode(cv2.imencode(".jpg", face_img)[1]).decode("utf-8")}',
                'name': student_name,
                'student_id': student_id,
                'confidence': confidence_percentage,
                'capture_time': capture_time,
                'prediction_time': prediction_time,
                'attendance_type': attendance_type
            }
        else:
            return jsonify({'error': 'Face recognition confidence is too low'})

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

    return jsonify(response)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html')
    return render_template('home.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        print(f"Login attempt: email={email}, password={password}")  # Kiểm tra email và mật khẩu

        # Lấy dữ liệu người dùng từ Firestore
        user_ref = firestore_db.collection('users').document(email)
        user_data = user_ref.get()

        if user_data.exists:
            user = user_data.to_dict()
            print(f"User data from Firestore: {user}")  # Kiểm tra dữ liệu lấy từ Firestore

            # Kiểm tra mật khẩu và trạng thái xác minh
            if check_password_hash(user['password'], password) and user.get('is_verified', False):
                session['username'] = email
                print(f"User {email} authenticated successfully.")  # Xác nhận thành công

                # Kiểm tra vai trò của người dùng
                role = user.get('role')
                print(f"User role: {role}")  # Kiểm tra vai trò
                if role == 'admin':
                    print("Redirecting to admin.html")
                    return redirect(url_for('admin'))  # Chuyển đến trang admin
                elif role == 'student':
                    print("Redirecting to student.html")
                    return redirect(url_for('student'))  # Chuyển đến trang student
                else:
                    print("Redirecting to index.html")
                    return redirect(url_for('index'))  # Chuyển đến trang index
            else:
                print("Invalid password or unverified email.")  # Mật khẩu sai hoặc email chưa xác minh
                flash('Email hoặc mật khẩu không hợp lệ, hoặc email chưa được xác minh.', 'error')
        else:
            print(f"User {email} does not exist in Firestore.")  # Email không tồn tại
            flash('Email không tồn tại.', 'error')

    return render_template('login.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        fullname = request.form.get('fullname')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')

        # Kiểm tra tính hợp lệ dữ liệu
        print(f"Fullname: {fullname}, Email: {email}, Password: {password}, Role: {role}")

        if not fullname or not email or not password or not role:
            flash('All fields are required.', 'error')
            return redirect(url_for('register'))

        if '@' not in email:
            flash('Invalid email address.', 'error')
            return redirect(url_for('register'))

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return redirect(url_for('register'))

        # Kiểm tra người dùng đã tồn tại chưa
        user_ref = firestore_db.collection('users').document(email)  # Sử dụng email làm ID
        print(f"Checking if user exists with email: {email}")
        if user_ref.get().exists:
            flash('User with this email already exists.', 'error')
            return redirect(url_for('register'))

        # Kiểm tra kết nối Firestore
        try:
            # Kiểm tra kết nối Firestore
            firestore_db.collection('users').get()
            print("Connected to Firestore")
        except Exception as e:
            print(f"Error connecting to Firestore: {e}")
            flash('Error connecting to database. Please try again later.', 'error')
            return redirect(url_for('register'))

        # Mã hóa mật khẩu trước khi lưu
        hashed_password = generate_password_hash(password)
        print(f"Hashed password: {hashed_password}")

        # Lưu thông tin người dùng vào Firestore
        user_data = {
            'email': email,
            'password': hashed_password,  # Mật khẩu đã mã hóa
            'role': role,
            'is_verified': True  # Không cần OTP nên mặc định là đã xác minh
        }
        try:
            user_ref.set(user_data)
            print("User data saved successfully to Firestore.")
        except Exception as e:
            print(f"Error saving user data to Firestore: {e}")
            flash('An error occurred while saving your data. Please try again later.', 'error')
            return redirect(url_for('register'))

        flash('Registration successful. You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = next((u for u in users_db.values() if u['email'] == email), None)

        if user:
            new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
            msg = Message('Your New Password', sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = f'Your new password is {new_password}. Please log in and change it as soon as possible.'
            mail.send(msg)

            username = next(u for u, d in users_db.items() if d['email'] == email)
            users_db[username]['password'] = new_password

            flash('A new password has been sent to your email.', 'success')
        else:
            flash('Email address not found.', 'error')

        return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html')
@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    # Kiểm tra người dùng đã đăng nhập chưa
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not current_password or not new_password or not confirm_password:
            flash('Tất cả các trường là bắt buộc.', 'error')
            return redirect(url_for('change_password'))

        if len(new_password) < 8:
            flash('Mật khẩu mới phải dài ít nhất 8 ký tự.', 'error')
            return redirect(url_for('change_password'))

        if new_password != confirm_password:
            flash('Mật khẩu mới và xác nhận mật khẩu không khớp.', 'error')
            return redirect(url_for('change_password'))

        # Lấy thông tin người dùng từ Firestore
        user_email = session['username']
        user_ref = firestore_db.collection('users').document(user_email)
        user_data = user_ref.get()

        if not user_data.exists:
            flash('Không tìm thấy người dùng.', 'error')
            return redirect(url_for('index'))

        user = user_data.to_dict()
        # Kiểm tra mật khẩu hiện tại
        if not check_password_hash(user['password'], current_password):
            flash('Mật khẩu hiện tại không đúng.', 'error')
            return redirect(url_for('change_password'))

        # Mã hóa mật khẩu mới
        hashed_new_password = generate_password_hash(new_password)

        # Cập nhật mật khẩu mới vào Firestore
        try:
            user_ref.update({'password': hashed_new_password})
            flash('Mật khẩu đã được thay đổi thành công.', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            flash('Có lỗi khi cập nhật mật khẩu. Vui lòng thử lại sau.', 'error')
            return redirect(url_for('change_password'))

    return render_template('change_password.html')
@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    # Kiểm tra người dùng đã đăng nhập chưa
    if 'username' not in session:
        return redirect(url_for('login'))

    # Lấy thông tin người dùng từ Firestore
    user_email = session['username']
    user_ref = firestore_db.collection('users').document(user_email)
    user_data = user_ref.get()

    if not user_data.exists:
        flash('Không tìm thấy người dùng.', 'error')
        return redirect(url_for('index'))

    user = user_data.to_dict()

    if request.method == 'POST':
        fullname = request.form.get('fullname')
        new_email = request.form.get('email')

        # Kiểm tra xem người dùng có thay đổi email hay không
        if new_email != user_email:
            flash('Không thể thay đổi email tại thời điểm này.', 'error')
            return redirect(url_for('edit_profile'))

        if not fullname:
            flash('Tên đầy đủ là bắt buộc.', 'error')
            return redirect(url_for('edit_profile'))

        # Cập nhật thông tin người dùng
        try:
            user_ref.update({
                'fullname': fullname
            })
            flash('Thông tin cá nhân đã được cập nhật thành công.', 'success')
            return redirect(url_for('profile'))  # Quay lại trang hồ sơ sau khi cập nhật
        except Exception as e:
            flash('Có lỗi khi cập nhật thông tin. Vui lòng thử lại sau.', 'error')
            return redirect(url_for('edit_profile'))

    return render_template('edit_profile.html', user=user)


@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    email = request.form['email']
    feedback = request.form['feedback']
    rating = request.form.get('rating')
    return 'Feedback received! Thank you!'

@app.route('/index')
def index():
    if 'username' in session:
        email = session['username']

        # Lấy thông tin người dùng từ Firestore
        user_ref = firestore_db.collection('users').document(email)
        user = user_ref.get()

        if user.exists:
            user_data = user.to_dict()
            fullname = user_data.get('fullname', 'N/A')
            return render_template('index.html', fullname=fullname, email=email)
        else:
            flash('Không tìm thấy thông tin người dùng.', 'error')
            return redirect(url_for('login'))
    else:
        return redirect(url_for('login'))
@app.route('/admin')
def admin():
    if 'username' not in session:
        flash("Bạn cần đăng nhập để truy cập trang quản trị.", "error")
        return redirect(url_for('login'))

    email = session['username']

    # Lấy thông tin user từ Firestore
    user_ref = firestore_db.collection('users').document(email)
    user_data = user_ref.get()

    if user_data.exists:
        user = user_data.to_dict()
        if user.get('role') != 'admin':
            flash("Bạn không có quyền truy cập trang này.", "error")
            return redirect(url_for('index'))
        return render_template('admin.html', admin_email=email)
    else:
        flash("Dữ liệu người dùng không tồn tại.", "error")
        return redirect(url_for('login'))
@app.route('/User_Mana', methods=['GET'])
def user_management_page():
    return render_template("User_Mana.html")

@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        users_ref = firestore_db.collection(USER_COLLECTION)
        users = [doc.to_dict() | {"id": doc.id} for doc in users_ref.stream()]
        return jsonify(users), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/users', methods=['POST'])
def add_user():
    try:
        user_data = request.json
        if not user_data:
            return jsonify({"error": "No user data provided"}), 400

        # Thêm người dùng vào Firestore
        new_user_ref = firestore_db.collection(USER_COLLECTION).add(user_data)
        return jsonify({"message": "User added successfully", "id": new_user_ref[1].id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        user_data = request.json
        if not user_data:
            return jsonify({"error": "No update data provided"}), 400

        # Cập nhật dữ liệu người dùng
        user_ref = firestore_db.collection(USER_COLLECTION).document(user_id)
        if not user_ref.get().exists:
            return jsonify({"error": "User not found"}), 404

        user_ref.update(user_data)
        return jsonify({"message": "User updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user_ref = firestore_db.collection(USER_COLLECTION).document(user_id)
        if not user_ref.get().exists:
            return jsonify({"error": "User not found"}), 404

        user_ref.delete()
        return jsonify({"message": "User deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/student')
def student():
    # Kiểm tra xem người dùng đã đăng nhập chưa
    if 'username' not in session:
        flash('Bạn cần đăng nhập để truy cập trang này.', 'error')
        return redirect(url_for('login'))

    # Lấy email từ session
    email = session['username']
    # Lấy dữ liệu người dùng từ Firestore
    user_ref = firestore_db.collection('users').document(email)
    user_data = user_ref.get()

    if user_data.exists:
        user = user_data.to_dict()
        role = user.get('role', '')

        # Kiểm tra vai trò, chỉ cho phép sinh viên truy cập
        if role == 'student':
            return render_template('student.html', student_email=email)
        else:
            flash('Bạn không có quyền truy cập vào trang này.', 'error')
            return redirect(url_for('login'))
    else:
        flash('Dữ liệu người dùng không tồn tại.', 'error')
        return redirect(url_for('login'))
@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        student_name = request.form.get('student_name')
        student_id = request.form.get('student_id')
        student_face = request.files.get('student_face')

        # Lưu ảnh khuôn mặt vào thư mục cụ thể
        face_filename = f"{student_id}_face.jpg"
        student_face.save(f"static/faces/{face_filename}")

        # Thêm thông tin học sinh vào cơ sở dữ liệu (ví dụ Firestore)
        student_ref = firestore_db.collection('students').document(student_id)
        student_ref.set({
            'name': student_name,
            'student_id': student_id,
            'face_image': face_filename
        })

        flash('Học sinh đã được thêm thành công!', 'success')
        return redirect(url_for('admin'))  # Chuyển đến trang admin sau khi thêm học sinh

    return render_template('add_student.html')
save_path = "C:/Users/admin/Desktop/video/anhsinhvien/"

@app.route('/save-image', methods=['POST'])
def save_image():
    data = request.get_json()
    image_data = data['image']
    file_name = data['fileName']
    image_path = os.path.join(save_path, file_name)

    try:
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_data.split(",")[1]))
        return jsonify({"message": "Image saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/user_info')
def user_info():
    if 'username' in session:
        email = session['username']

        # Lấy thông tin người dùng từ Firestore
        user_ref = firestore_db.collection('users').document(email)
        user = user_ref.get()

        if user.exists:
            user_data = user.to_dict()
            fullname = user_data.get('fullname', 'N/A')
            role = user_data.get('role', 'N/A')
            return render_template('user_info.html', fullname=fullname, email=email, role=role)
        else:
            flash('Không tìm thấy thông tin người dùng.', 'error')
            return redirect(url_for('login'))
    else:
        return redirect(url_for('login'))
@app.route('/profile')
def profile():
    # Kiểm tra người dùng đã đăng nhập chưa
    if 'username' not in session:
        return redirect(url_for('login'))

    user_email = session['username']
    # Lấy thông tin người dùng từ Firestore
    user_ref = firestore_db.collection('users').document(user_email)
    user_data = user_ref.get()

    if user_data.exists:
        user = user_data.to_dict()
        return render_template('profile.html', user=user)
    else:
        flash('Không tìm thấy thông tin người dùng.', 'error')
        return redirect(url_for('index'))


@app.route('/history')
def history():
    # Lấy dữ liệu từ Firebase Realtime Database
    ref = db.reference('attendance')  # Đường dẫn đến dữ liệu 'attendance' trong Firebase
    attendance_data = ref.get()  # Lấy toàn bộ dữ liệu dưới "attendance"

    # Chuyển dữ liệu về định dạng danh sách cho dễ sử dụng trong template
    if attendance_data:
        formatted_data = []
        for student_id, records in attendance_data.items():
            for capture_time, record in records.items():
                formatted_data.append(record)

        # Trả về trang HTML và truyền dữ liệu
        return render_template('history.html', attendance_data=formatted_data)
    else:
        return render_template('history.html', attendance_data=None)


@app.route('/class_information', methods=['GET', 'POST'])
def class_information():
    try:
        if request.method == 'POST':
            # Nhận dữ liệu lớp học từ POST request
            new_class = request.get_json()
            # Kiểm tra xem các trường cần thiết có trong dữ liệu không
            required_fields = ["Mã Lớp", "Tên Lớp", "Khoa", "Giảng Viên", "Năm Học", "Thời Gian Học", "Phòng Học"]
            for field in required_fields:
                if field not in new_class:
                    return jsonify({"error": f"Missing field: {field}"}), 400
            # Lưu lớp học vào Firebase
            class_ref.push(new_class)  # Thêm lớp học mới vào Firebase

            return jsonify(new_class), 201  # Trả lại lớp học đã thêm

        else:
            # Lấy tất cả dữ liệu lớp học từ Firebase
            class_data = class_ref.get()

            # Nếu không có dữ liệu
            if not class_data:
                return render_template('class_information.html', error_message="No class data available.")

            # Chuyển dữ liệu lớp học thành danh sách để dễ render trên HTML
            class_data_list = [value for value in class_data.values()]

            return render_template('class_information.html', class_data=class_data_list)

    except Exception as e:
        print("Error in class_information:", e)
        return jsonify({"error": "Failed to process the data"}), 500
# Route để hiển thị lớp học và thêm mới lớp học từ HTML form
@app.route('/add_class', methods=['GET', 'POST'])
def add_class():
    if request.method == 'POST':
        # Nhận thông tin từ form
        new_class = {
            "Mã Lớp": request.form['ma_lop'],
            "Tên Lớp": request.form['ten_lop'],
            "Khoa": request.form['khoa'],
            "Giảng Viên": request.form['giang_vien'],
            "Năm Học": request.form['nam_hoc'],
            "Thời Gian Học": request.form['thoi_gian_hoc'],
            "Phòng Học": request.form['phong_hoc']
        }

        # Lưu lớp học vào Firebase
        class_ref.push(new_class)

        return jsonify({"message": "Class added successfully!"}), 201

    return render_template('add_class.html')


@app.route('/attendance_history', methods=['GET'])
def attendance_history():
    # Lấy dữ liệu từ Firebase Realtime Database
    ref = db.reference('attendance')  # Đường dẫn đến dữ liệu 'attendance' trong Firebase
    attendance_data = ref.get()  # Lấy toàn bộ dữ liệu dưới "attendance"

    # Kiểm tra nếu có dữ liệu
    if attendance_data:
        # Lấy các tham số lọc từ request
        name_filter = request.args.get('name', '').lower()
        student_id_filter = request.args.get('student_id', '').lower()
        attendance_type_filter = request.args.get('attendance_type', '').lower()
        start_date_filter = request.args.get('start_date', '')
        end_date_filter = request.args.get('end_date', '')

        # Chuẩn bị danh sách kết quả sau khi lọc
        filtered_data = []

        for student_id, records in attendance_data.items():
            for capture_time, record in records.items():
                # Áp dụng bộ lọc
                record_date = datetime.strptime(record['Capture Time'],
                                                '%Y-%m-%d %H:%M:%S') if 'Capture Time' in record else None
                if name_filter and name_filter not in record['Name'].lower():
                    continue
                if student_id_filter and student_id_filter not in record['Student ID'].lower():
                    continue
                if attendance_type_filter and attendance_type_filter != record['Attendance Type'].lower():
                    continue
                if start_date_filter and record_date and record_date < datetime.strptime(start_date_filter, '%Y-%m-%d'):
                    continue
                if end_date_filter and record_date and record_date > datetime.strptime(end_date_filter, '%Y-%m-%d'):
                    continue

                filtered_data.append(record)

        return render_template('attendance_history.html', attendance_data=filtered_data)

    else:
        return render_template('attendance_history.html', attendance_data=None)
@app.route('/get_classes', methods=['GET'])
def get_classes():
    try:
        # Lấy thông tin các lớp học từ Firebase
        classes = ref.get()
        # Nếu không có lớp học nào, trả về thông báo lỗi
        if not classes:
            return jsonify({"error": "No classes found"}), 404
        # Trả về danh sách lớp học
        return jsonify(classes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/class_information', methods=['GET'])
def get_class_information():
    # Lấy dữ liệu từ Firebase
    class_data = class_ref.get()

    # Trả về dữ liệu dưới dạng JSON
    return jsonify(class_data)
if __name__ == '__main__':
    app.run(debug=True)
