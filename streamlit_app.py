import streamlit as st
import cv2
import numpy as np
import sqlite3
import insightface
from PIL import Image
import time
import gc
from datetime import datetime

# Khởi tạo cơ sở dữ liệu
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY, date TEXT, time TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (session_id INTEGER, student_id INTEGER, status TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

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
def find_closest_match(embedding, ids, names, embeddings, threshold=1.0):
    if not embeddings:
        return None, None
    distances = [np.linalg.norm(embedding - emb) for emb in embeddings]
    min_distance = min(distances)
    if min_distance < threshold:
        index = distances.index(min_distance)
        return ids[index], names[index]
    return None, None

# Tạo phiên điểm danh mới
def create_new_session():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO sessions (date, time) VALUES (date('now'), time('now'))")
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

# Ghi nhận điểm danh
def mark_attendance(session_id, student_id, timestamp):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO attendance (session_id, student_id, status, timestamp) VALUES (?, ?, 'present', ?)",
              (session_id, student_id, timestamp))
    conn.commit()
    conn.close()

# Lấy danh sách buổi học
def get_sessions():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT id, date, time FROM sessions")
    sessions = c.fetchall()
    conn.close()
    return sessions

# CSS để làm giao diện chuyên nghiệpового
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
st.sidebar.title("Điều Hướng")
page = st.sidebar.radio("Chọn Trang", ["Đăng Ký Sinh Viên", "Điểm Danh", "Xem Điểm Danh"])

if page == "Đăng Ký Sinh Viên":
    st.header("Đăng Ký Sinh Viên")
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
                conn = sqlite3.connect('attendance.db')
                c = conn.cursor()
                c.execute("INSERT INTO students (name, embedding) VALUES (?, ?)", (name, embedding.tobytes()))
                conn.commit()
                conn.close()
                st.success(f"Đã đăng ký sinh viên {name} thành công!")
                ids, names, embeddings = load_embeddings()
            else:
                st.error("Không phát hiện khuôn mặt hoặc có nhiều khuôn mặt. Vui lòng chụp lại với chỉ một khuôn mặt.")

elif page == "Điểm Danh":
    st.header("Điểm Danh Buổi Học")
    
    # Nút tạo buổi học mới
    if st.button("Tạo Buổi Học Mới"):
        session_id = create_new_session()
        st.success(f"Đã tạo buổi học mới với ID: {session_id}")
    
    # Lấy và hiển thị danh sách buổi học
    sessions = get_sessions()
    session_options = [f"Buổi {s[0]} ngày {s[1]} lúc {s[2]}" for s in sessions]
    selected_session = st.selectbox("Chọn Buổi Học", session_options)
    
    if selected_session:
        session_id = int(selected_session.split()[1])
        if st.button("Bắt Đầu Điểm Danh"):
            st.info("Đang điểm danh... Nhấn 'Dừng Điểm Danh' để kết thúc.")
            cap = cv2.VideoCapture(0)  # Thay index 1 thành 0
            if not cap.isOpened():
                st.error("Không thể mở camera. Vui lòng kiểm tra kết nối hoặc index camera.")
                st.stop()
            recognized_students = set()
            placeholder = st.empty()
            stop_button = st.button("Dừng Điểm Danh")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                faces = recognizer.app.get(frame)
                for face in faces:
                    embedding = face.embedding
                    student_id, student_name = find_closest_match(embedding, ids, names, embeddings)
                    if student_id is not None and student_id not in recognized_students:
                        recognized_students.add(student_id)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        mark_attendance(session_id, student_id, timestamp)
                        placeholder.success(f"Đã điểm danh: {student_name} lúc {timestamp}")
                if stop_button:
                    break
            cap.release()
            gc.collect()
            st.success("Đã dừng quá trình điểm danh.")
            
elif page == "Xem Điểm Danh":
    st.header("Xem Danh Sách Điểm Danh")
    sessions = get_sessions()
    session_options = [f"Buổi {s[0]} ngày {s[1]} lúc {s[2]}" for s in sessions]
    selected_session = st.selectbox("Chọn Buổi", session_options)
    if selected_session:
        session_id = int(selected_session.split()[1])
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("SELECT s.name, a.timestamp FROM students s JOIN attendance a ON s.id = a.student_id WHERE a.session_id = ? AND a.status = 'present'",
                  (session_id,))
        present_students = c.fetchall()
        st.subheader("Danh sách sinh viên có mặt:")
        if present_students:
            for student in present_students:
                st.write(f"- {student[0]} lúc {student[1]}")
        else:
            st.write("Không có sinh viên nào được ghi nhận.")
        conn.close()
