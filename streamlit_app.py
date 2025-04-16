import streamlit as st
import cv2
import numpy as np
import sqlite3
import insightface
from PIL import Image
import time
import gc
from datetime import datetime
import os

# Khởi tạo cơ sở dữ liệu
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB, image_path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY, date TEXT, time TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (session_id INTEGER, student_id INTEGER, status TEXT, timestamp TEXT)''')
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
def find_closest_match(embedding, ids, names, embeddings, threshold=20):
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
st.title("Ứng Dụng Điểm Danh Sinh Viên")
page = st.sidebar.radio("Chọn Chức năng", ["Đăng Ký Sinh Viên", "Điểm Danh", "Xem Sinh Viên"])

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
                image_path = f"student_images/{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
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
        st.subheader("Chụp hoặc tải ảnh để điểm danh")
        
        # Widget để chụp ảnh hoặc tải lên ảnh
        image_file = st.camera_input("Chụp ảnh để điểm danh")  # Cho phép chụp ảnh
        
        # Tùy chọn tải lên ảnh nếu không muốn dùng camera
        uploaded_file = st.file_uploader("Hoặc tải lên ảnh để điểm danh", type=["jpg", "png", "jpeg"])
        
        # Xử lý ảnh từ camera hoặc file tải lên
        if image_file is not None or uploaded_file is not None:
            # Ưu tiên ảnh từ camera nếu có, nếu không thì dùng ảnh tải lên
            file_to_process = image_file if image_file is not None else uploaded_file
            image = Image.open(file_to_process)
            img_array = np.array(image)
            faces = recognizer.app.get(img_array)
            
            if len(faces) == 1:
                embedding = faces[0].embedding
                student_id, student_name = find_closest_match(embedding, ids, names, embeddings)
                if student_id is not None:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    mark_attendance(session_id, student_id, timestamp)
                    st.success(f"Đã điểm danh: {student_name} lúc {timestamp}")
                else:
                    st.error("Không nhận diện được sinh viên trong ảnh.")
            else:
                st.error("Ảnh không chứa đúng một khuôn mặt. Vui lòng chụp hoặc tải lại.")

elif page == "Xem Sinh Viên":
    view_students_page()
