import streamlit as st
import cv2
import numpy as np
import sqlite3
import insightface
from PIL import Image
import time
import gc
from datetime import datetime, timedelta
import os
import pytz

# Thiết lập múi giờ Việt Nam (UTC+7)
tz = pytz.timezone('Asia/Ho_Chi_Minh')

# Khởi tạo cơ sở dữ liệu
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB, image_path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY, class_name TEXT, session_date TEXT, session_day TEXT, start_time TEXT, end_time TEXT, max_attendance_score INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (session_id INTEGER, student_id INTEGER, status TEXT, timestamp TEXT, attendance_score INTEGER)''')
    conn.commit()
    conn.close()

init_db()

# Kết nối đến cơ sở dữ liệu
def get_db_connection():
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row
    return conn

# Lấy danh sách sinh viên từ cơ sở dữ liệu
def get_students():
    conn = get_db_connection()
    students = conn.execute('SELECT id, name, image_path FROM students').fetchall()
    conn.close()
    return students

# Trang xem danh sách sinh viên
def view_students_page():
    st.header("Danh Sách Sinh Viên Đã Đăng Ký")
    
    students = get_students()
    
    if not students:
        st.info("Chưa có sinh viên nào được đăng ký.")
        return
    
    for student in students:
        st.subheader(f"Sinh viên: {student['name']}")
        image_path = student['image_path']
        
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=f"Hình ảnh của {student['name']}", use_column_width=True)
        else:
            st.warning(f"Không tìm thấy hình ảnh cho sinh viên {student['name']}.")
        
        st.write(f"ID: {student['id']}")
        st.write("---")

# Lớp nhận diện khuôn mặt
class FaceRecognizer:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis()
        self.app.prepare(ctx_id=0)  # Sử dụng CPU

    def get_embedding(self, image):
        faces = self.app.get(image)
        if len(faces) == 1:
            return faces[0].embedding
        return None

# Cache mô hình để tránh tải lại
@st.cache_resource
def get_recognizer():
    return FaceRecognizer()

recognizer = get_recognizer()

# Tải embedding của sinh viên
def load_embeddings():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT id, name, embedding FROM students")
    students = c.fetchall()
    conn.close()
    embeddings = []
    ids = []
    names = []
    for student in students:
        ids.append(student[0])
        names.append(student[1])
        embeddings.append(np.frombuffer(student[2], dtype=np.float32))
    return ids, names, embeddings

ids, names, embeddings = load_embeddings()

# Tìm sinh viên khớp nhất
def find_closest_match(embedding, ids, names, embeddings, threshold=20.0):
    if not embeddings:
        return None, None
    distances = [np.linalg.norm(embedding - emb) for emb in embeddings]
    min_distance = min(distances)
    if min_distance < threshold:
        index = distances.index(min_distance)
        return ids[index], names[index]
    return None, None

# Tạo buổi thực tập
def create_new_session(class_name, session_date, session_day, start_time, end_time, max_attendance_score):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO sessions (class_name, session_date, session_day, start_time, end_time, max_attendance_score) VALUES (?, ?, ?, ?, ?, ?)",
              (class_name, session_date, session_day, start_time, end_time, max_attendance_score))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

# Ghi nhận điểm danh
def mark_attendance(session_id, student_id, timestamp, attendance_score):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO attendance (session_id, student_id, status, timestamp, attendance_score) VALUES (?, ?, 'present', ?, ?)",
              (session_id, student_id, timestamp, attendance_score))
    conn.commit()
    conn.close()

# Lấy danh sách buổi thực tập
def get_sessions():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT id, class_name, session_date, session_day, start_time, end_time, max_attendance_score FROM sessions")
    sessions = c.fetchall()
    conn.close()
    return sessions

# Lấy thông tin buổi thực tập
def get_session_info(session_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session = c.fetchone()
    conn.close()
    return session

# Lấy danh sách sinh viên đã điểm danh trong buổi thực tập
def get_attendance_list(session_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT s.name, a.timestamp, a.attendance_score FROM students s JOIN attendance a ON s.id = a.student_id WHERE a.session_id = ? AND a.status = 'present'",
              (session_id,))
    attendance_list = c.fetchall()
    conn.close()
    return attendance_list

# CSS để làm giao diện chuyên nghiệp
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    background-color: #f4f4f9;
}
h1, h2 {
    color: #2c3e50;
    text-align: center;
}
.stButton>button {
    background-color: #3498db;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
}
.stTextInput>div>input {
    border: 1px solid #3498db;
    border-radius: 5px;
    padding: 8px;
}
.sidebar .sidebar-content {
    background-color: #ecf0f1;
}
</style>
""", unsafe_allow_html=True)

# Ứng dụng Streamlit
st.title("Ứng Dụng Điểm Danh Thực Tập Hóa Sinh - Bộ môn Hóa Sinh")
page = st.sidebar.radio("Chọn Chức năng", ["Đăng Ký Sinh Viên", "Tạo Buổi Thực Tập", "Điểm Danh", "Xem Sinh Viên", "Xem Điểm Danh"])

if page == "Đăng Ký Sinh Viên":
    st.header("Đăng Ký Sinh Viên Mới")
    col1, col2 = st.columns([2, 1])
    with col1:
        image_file = st.camera_input("Chụp ảnh sinh viên")
    with col2:
        name = st.text_input("Tên Sinh Viên")
        if st.button("Đăng Ký") and image_file is not None and name:
            image = Image.open(image_file)
            img_array = np.array(image)
            embedding = recognizer.get_embedding(img_array)
            if embedding is not None:
                # Tạo thư mục lưu hình ảnh nếu chưa có
                if not os.path.exists('student_images'):
                    os.makedirs('student_images')
                
                # Lưu hình ảnh
                image_path = f"student_images/{name}_{datetime.now(tz).strftime('%Y%m%d%H%M%S')}.jpg"
                image.save(image_path)
                
                # Lưu vào cơ sở dữ liệu
                conn = sqlite3.connect('attendance.db')
                c = conn.cursor()
                c.execute("INSERT INTO students (name, embedding, image_path) VALUES (?, ?, ?)", 
                          (name, embedding.tobytes(), image_path))
                conn.commit()
                conn.close()
                st.success(f"Đã đăng ký sinh viên {name} thành công!")
                ids, names, embeddings = load_embeddings()
            else:
                st.error("Không phát hiện khuôn mặt hoặc có nhiều khuôn mặt. Vui lòng chụp lại với chỉ một khuôn mặt.")

elif page == "Tạo Buổi Thực Tập":
    st.header("Tạo Buổi Thực Tập Mới")
    
    # Lấy ngày hiện tại theo múi giờ Việt Nam
    today = datetime.now(tz).date()
    today_str = today.strftime("%Y-%m-%d")
    day_of_week = today.strftime("%A")  # Lấy thứ trong tuần
    
    class_name = st.text_input("Khối lớp thực tập (ví dụ: Lớp 10A)", "")
    session_date = st.date_input("Ngày thực tập", value=today)
    session_day = st.selectbox("Thứ trong tuần", ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"], index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week))
    start_time = st.time_input("Giờ bắt đầu đánh giá điểm chuyên cần", value=datetime.now(tz).time())
    end_time = st.time_input("Giờ kết thúc đánh giá điểm chuyên cần", value=(datetime.now(tz) + timedelta(hours=1)).time())
    max_attendance_score = st.number_input("Điểm chuyên cần tối đa (1-10)", min_value=1, max_value=10, value=10)
    
    if st.button("Tạo Buổi Thực Tập"):
        session_id = create_new_session(class_name, session_date.strftime("%Y-%m-%d"), session_day, start_time.strftime("%H:%M"), end_time.strftime("%H:%M"), max_attendance_score)
        st.success(f"Đã tạo buổi thực tập mới với ID: {session_id}")

elif page == "Điểm Danh":
    st.header("Điểm Danh Buổi Thực Tập")
    
    sessions = get_sessions()
    session_options = [f"Buổi {s[0]} - {s[1]} - {s[2]} ({s[3]})" for s in sessions]
    selected_session = st.selectbox("Chọn Buổi Thực Tập", session_options)
    
    if selected_session:
        session_id = int(selected_session.split()[1])
        session_info = get_session_info(session_id)
        st.subheader(f"Điểm danh cho buổi thực tập: {session_info['class_name']} - {session_info['session_date']} ({session_info['session_day']})")
        
        image_file = st.camera_input("Chụp ảnh để điểm danh")
        uploaded_file = st.file_uploader("Hoặc tải lên ảnh để điểm danh", type=["jpg", "png", "jpeg"])
        
        if image_file is not None or uploaded_file is not None:
            file_to_process = image_file if image_file is not None else uploaded_file
            image = Image.open(file_to_process)
            img_array = np.array(image)
            faces = recognizer.app.get(img_array)
            
            if len(faces) == 1:
                embedding = faces[0].embedding
                student_id, student_name = find_closest_match(embedding, ids, names, embeddings)
                if student_id is not None:
                    # Lấy thời gian hiện tại theo múi giờ Việt Nam
                    now = datetime.now(tz)
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Lấy khung giờ đánh giá
                    start_time = datetime.strptime(f"{session_info['session_date']} {session_info['start_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                    end_time = datetime.strptime(f"{session_info['session_date']} {session_info['end_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                    
                    # Tính điểm chuyên cần
                    if start_time <= now <= end_time:
                        attendance_score = session_info['max_attendance_score']
                    else:
                        attendance_score = 0
                    
                    # Lưu điểm danh
                    mark_attendance(session_id, student_id, timestamp, attendance_score)
                    
                    # Hiển thị thông tin sinh viên và hình ảnh gốc
                    conn = sqlite3.connect('attendance.db')
                    c = conn.cursor()
                    c.execute("SELECT image_path FROM students WHERE id = ?", (student_id,))
                    image_path = c.fetchone()[0]
                    conn.close()
                    
                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption=f"Hình ảnh gốc của {student_name}", width=150)
                    else:
                        st.warning(f"Không tìm thấy hình ảnh gốc cho sinh viên {student_name}.")
                    
                    st.success(f"Đã điểm danh: {student_name} lúc {timestamp} - Điểm chuyên cần: {attendance_score}")
                else:
                    st.error("Không nhận diện được sinh viên trong ảnh.")
            else:
                st.error("Ảnh không chứa đúng một khuôn mặt. Vui lòng chụp hoặc tải lại.")

elif page == "Xem Sinh Viên":
    view_students_page()

elif page == "Xem Điểm Danh":
    st.header("Xem Danh Sách Điểm Danh")
    sessions = get_sessions()
    session_options = [f"Buổi {s[0]} - {s[1]} - {s[2]} ({s[3]})" for s in sessions]
    selected_session = st.selectbox("Chọn Buổi Thực Tập", session_options)
    
    if selected_session:
        session_id = int(selected_session.split()[1])
        session_info = get_session_info(session_id)
        st.subheader(f"Danh sách sinh viên đã điểm danh cho buổi thực tập: {session_info['class_name']} - {session_info['session_date']} ({session_info['session_day']})")
        
        attendance_list = get_attendance_list(session_id)
        
        if attendance_list:
            for student in attendance_list:
                st.write(f"- {student[0]} - Giờ có mặt: {student[1]} - Điểm chuyên cần: {student[2]}")
        else:
            st.write("Không có sinh viên nào được ghi nhận.")
