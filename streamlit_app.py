import streamlit as st
import cv2
import numpy as np
import sqlite3
import insightface
from PIL import Image
import time
import gc
import base64
from datetime import datetime, timedelta
import os
import pytz
import pandas as pd
from io import BytesIO
from camera_input_live import camera_input_live
import zipfile

# Thiết lập múi giờ Việt Nam (UTC+7)
tz = pytz.timezone('Asia/Ho_Chi_Minh')

# Khởi tạo cơ sở dữ liệu
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (record_id INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT, name TEXT, embedding BLOB, image_path TEXT, session_id INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY, class_name TEXT, session_date TEXT, session_day TEXT, start_time TEXT, end_time TEXT, max_attendance_score INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (session_id INTEGER, student_id TEXT, status TEXT, timestamp TEXT, attendance_score INTEGER, note TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Kết nối đến cơ sở dữ liệu
def get_db_connection():
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row
    return conn

# Lấy danh sách sinh viên theo session_id
def get_students_by_session(session_id):
    conn = get_db_connection()
    students = conn.execute('SELECT record_id, id, name, image_path, session_id FROM students WHERE session_id = ?', (session_id,)).fetchall()
    conn.close()
    return students

# Lấy danh sách khối thực tập
def get_sessions_list():
    conn = get_db_connection()
    sessions = conn.execute('SELECT id, class_name, session_date, session_day FROM sessions').fetchall()
    conn.close()
    return sessions

# Lấy tên sinh viên theo student_id
def get_student_name(student_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT name FROM students WHERE id = ? LIMIT 1", (student_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return None

# Lấy hình ảnh của sinh viên theo record_id
def get_student_image(record_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT image_path FROM students WHERE record_id = ?", (record_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return None

# Trang xem danh sách sinh viên
def view_students_page():
    st.header("Danh Sách Sinh Viên Đã Đăng Ký")
    
    sessions = get_sessions_list()
    if not sessions:
        st.info("Chưa có khối thực tập nào. Vui lòng tạo khối thực tập trước.")
        return
    
    session_options = [f"{s['class_name']} - {s['session_date']} ({s['session_day']})" for s in sessions]
    selected_session = st.selectbox("Chọn Khối Thực Tập", session_options)
    session_id = sessions[session_options.index(selected_session)]['id']
    
    students = get_students_by_session(session_id)
    
    if not students:
        st.info("Chưa có sinh viên nào được đăng ký cho khối thực tập này.")
        return
    
    df = pd.DataFrame(students, columns=['record_id', 'id', 'name', 'image_path', 'session_id'])
    st.dataframe(df[['record_id', 'id', 'name']])
    
    selected_record_id = st.selectbox("Chọn bản ghi để xem hình ảnh", df['record_id'])
    selected_student = df[df['record_id'] == selected_record_id].iloc[0]
    
    image_path = selected_student['image_path']
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=f"Hình ảnh của {selected_student['name']} (MSSV: {selected_student['id']})", use_container_width=True)
        
        # Thêm nút tải ảnh
        with open(image_path, "rb") as file:
            btn = st.download_button(
                label="Tải ảnh về máy",
                data=file,
                file_name=f"{selected_student['id']}_{selected_student['name']}.jpg",
                mime="image/jpeg"
            )
    else:
        st.warning(f"Không tìm thấy hình ảnh cho bản ghi {selected_record_id}.")
    
    if st.button("Xóa Bản Ghi Này"):
        conn = get_db_connection()
        conn.execute("DELETE FROM students WHERE record_id = ?", (selected_record_id,))
        conn.commit()
        conn.close()
        st.success(f"Đã xóa bản ghi {selected_record_id}.")
        st.rerun()
    
    if st.button("Tải về Danh Sách Sinh Viên (Excel)"):
        df_to_export = df[['record_id', 'id', 'name']]
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_to_export.to_excel(writer, index=False, sheet_name='Sinh Viên')
        excel_data = output.getvalue()
        st.download_button(
            label="Tải về file Excel",
            data=excel_data,
            file_name="danh_sach_sinh_vien.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Thêm nút tải xuống tất cả ảnh của buổi thực tập
    if st.button("Tải về tất cả ảnh Sinh Viên của buổi thực tập"):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for student in students:
                image_path = student['image_path']
                if image_path and os.path.exists(image_path):
                    zip_file.write(image_path, os.path.basename(image_path))
        zip_buffer.seek(0)
        st.download_button(
            label="Tải xuống file zip",
            data=zip_buffer,
            file_name=f"images_session_{session_id}.zip",
            mime="application/zip"
            )

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

# Tải embedding của sinh viên theo session_id
def load_embeddings_by_session(session_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT record_id, id, name, embedding FROM students WHERE session_id = ?", (session_id,))
    students = c.fetchall()
    conn.close()
    embeddings = []
    record_ids = []
    ids = []
    names = []
    for student in students:
        record_ids.append(student[0])
        ids.append(student[1])
        names.append(student[2])
        embeddings.append(np.frombuffer(student[3], dtype=np.float32))
    return record_ids, ids, names, embeddings

# Tìm sinh viên khớp nhất
def find_closest_match(embedding, record_ids, ids, names, embeddings, threshold=20.0):
    if not embeddings:
        return None, None, None
    student_embeddings = {}
    for record_id, student_id, name, emb in zip(record_ids, ids, names, embeddings):
        if student_id not in student_embeddings:
            student_embeddings[student_id] = []
        student_embeddings[student_id].append((record_id, emb))
    
    min_distances = {}
    for student_id, emb_list in student_embeddings.items():
        distances = [np.linalg.norm(embedding - emb[1]) for emb in emb_list]
        min_distance = min(distances)
        min_distances[student_id] = min_distance
    
    if min_distances:
        closest_student_id = min(min_distances, key=min_distances.get)
        min_distance = min_distances[closest_student_id]
        if min_distance < threshold:
            emb_list = student_embeddings[closest_student_id]
            distances = [np.linalg.norm(embedding - emb[1]) for emb in emb_list]
            index = distances.index(min_distance)
            record_id = emb_list[index][0]
            name = names[record_ids.index(record_id)]
            return record_id, closest_student_id, name
    return None, None, None

# Kiểm tra xem sinh viên đã được điểm danh trong buổi thực tập chưa
def check_attendance(session_id, student_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE session_id = ? AND student_id = ?", (session_id, student_id))
    result = c.fetchone()
    conn.close()
    return result is not None

# Ghi nhận điểm danh
def mark_attendance(session_id, student_id, timestamp, attendance_score, note):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO attendance (session_id, student_id, status, timestamp, attendance_score, note) VALUES (?, ?, 'present', ?, ?, ?)",
              (session_id, student_id, timestamp, attendance_score, note))
    conn.commit()
    conn.close()

# Tạo buổi thực tập
def create_new_session(class_name, session_date, session_day, start_time, end_time, max_attendance_score):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT id FROM sessions WHERE class_name = ? AND session_date = ?", (class_name, session_date))
    existing_session = c.fetchone()
    if existing_session:
        st.error(f"Khối thực tập '{class_name}' vào ngày '{session_date}' đã tồn tại.")
        return None
    c.execute("INSERT INTO sessions (class_name, session_date, session_day, start_time, end_time, max_attendance_score) VALUES (?, ?, ?, ?, ?, ?)",
              (class_name, session_date, session_day, start_time, end_time, max_attendance_score))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

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
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session = c.fetchone()
    conn.close()
    return session

# Lấy danh sách sinh viên đã điểm danh trong buổi thực tập
def get_attendance_list(session_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("""
        SELECT a.student_id, s.name, a.timestamp, a.attendance_score, a.note, ses.class_name, ses.session_date, ses.session_day, ses.start_time, ses.end_time
        FROM attendance a
        JOIN (SELECT id, MAX(name) as name FROM students GROUP BY id) s ON a.student_id = s.id
        JOIN sessions ses ON a.session_id = ses.id
        WHERE a.session_id = ? AND a.status = 'present'
    """, (session_id,))
    attendance_list = c.fetchall()
    conn.close()
    return attendance_list

# Hàm chuyển đổi thứ sang tiếng Việt
def get_vietnamese_day(day):
    days = {
        "Monday": "Thứ Hai",
        "Tuesday": "Thứ Ba",
        "Wednesday": "Thứ Tư",
        "Thursday": "Thứ Năm",
        "Friday": "Thứ Sáu",
        "Saturday": "Thứ Bảy",
        "Sunday": "Chủ Nhật"
    }
    return days.get(day, day)

# Đường dẫn đến hình nền
background_path = "image.jpg"

# Đọc và mã hóa hình ảnh
with open(background_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# CSS tùy chỉnh với palette màu xanh chuyên nghiệp
css = f"""
<style>
    :root {{
        --primary: #1E3A8A;
        --secondary: #3B82F6;
        --accent: #60A5FA;
        --light: #F0F4F8;
        --dark: #111827;
    }}

    .stApp {{
        position: relative;
        min-height: 100vh;
        background: var(--light); /* Thêm màu nền dự phòng */
        z-index: 0; /* Thêm z-index cho lớp cha */
    }}

    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url(data:image/jpeg;base64,{encoded_string});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        filter: blur(1px) brightness(0.85);
        opacity: 0.3;
        display: block !important;
        z-index: -1;
    }}

    .main .block-container {{
        background: rgba(255, 255, 255, 0.92) !important;
        backdrop-filter: blur(6px);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        padding: 2rem !important;
        padding-bottom: 80px !important;
    }}

    /* Các phần CSS khác giữ nguyên */
    h1 {{
        font-family: 'Segoe UI', sans-serif;
        color: var(--primary) !important;
        border-bottom: 3px solid var(--secondary);
        padding-bottom: 0.5rem;
        font-weight: 600 !important;
        text-align: center !important;
    }}

    .stButton > button {{
        background: var(--secondary) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        border: 1px solid var(--primary) !important;
        transition: all 0.2s ease-in-out !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(28, 58, 138, 0.1) !important;
    }}

    .stButton > button:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3) !important;
        background: var(--primary) !important;
    }}

    .stTextInput input {{
        border: 2px solid var(--accent) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }}

    .stDataFrame {{
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }}

    .stProgress > div > div {{
        background-color: var(--secondary) !important;
    }}
    
    /* Thêm style cho footer */
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--primary);
        color: white;
        text-align: center;
        padding: 15px 0;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        z-index: 1000;
    }}
    
/* CSS cho sidebar radio buttons */
.st-eb {{
        padding: 8px !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }}

.st-c7:hover:not(:disabled) {{
    background-color: var(--accent) !important;
}}

.st-da {{
    background-color: var(--secondary) !important;
}}

/* Style cho cảnh báo */
.stAlert {{
    background: rgba(59, 130, 246, 0.15) !important;
    border: 1px solid var(--secondary) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}}

.stAlert .st-emotion-cache-1m6dp5d {{
    color: var(--primary) !important;
}}

/* Style cho header */
.stMarkdown h2 {{
    color: var(--primary) !important;
    font-family: 'Segoe UI', sans-serif !important;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.3em !important;
    margin-bottom: 1em !important;
    font-weight: 650 !important;
}}

/* Style cho selectbox */
.stSelectbox > div > div {{
    border: 2px solid var(--accent) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}}

.stSelectbox > div > div:hover {{
    border-color: var(--secondary) !important;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.3) !important;
}}

.st-ef {{
    background-color: var(--light) !important;
    border: 1px solid var(--accent) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}}

/* Style cho text thông thường */
.main .block-container p {{
    color: var(--dark) !important;
    font-family: 'Segoe UI', sans-serif;
    line-height: 1.6;
    font-size: 1rem;
}}

/* Style cho code blocks */
.stCodeBlock pre {{
    background: var(--light) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}}

/* Hiệu ứng tổng thể cho các widget */
.st-bh, .st-c2, .st-c3, .st-c4 {{
    transition: all 0.3s ease-in-out !important;
}}

/* Style cho markdown tables */
.stMarkdown table {{
    border-collapse: collapse;
    margin: 1rem 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}}

.stMarkdown th {{
    background-color: var(--primary) !important;
    color: white !important;
    padding: 12px !important;
}}

.stMarkdown td {{
    background-color: rgba(240, 244, 248, 0.8) !important;
    padding: 10px !important;
    border-bottom: 1px solid var(--accent) !important;
}}

.sidebar .sidebar-content {{
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(8px);
        border-right: 1px solid rgba(0, 0, 0, 0.05);
    }}

</style>
"""

# Thêm HTML cho footer
footer = """
<div class="footer">
    Bộ môn Hóa Sinh - Trường ĐHYD Tp.Hồ Chí Minh - 2025
</div>
"""

# Hiển thị footer
#html(footer, height=60)

# Áp dụng CSS
st.markdown(css, unsafe_allow_html=True)
# Áp dụng HTML
st.markdown(footer, unsafe_allow_html=True)

# Ứng dụng Streamlit
st.title("Ứng dụng Quản lý Điểm chuyên cần Sinh viên")

# Initialize session state for navigation
if 'navigate_to' not in st.session_state:
    st.session_state['navigate_to'] = None

# Sidebar navigation
page = st.sidebar.radio("Chọn Chức năng", ["Đăng Ký Sinh Viên", "Tạo Buổi Thực Tập", "Điểm Danh", "Xem Sinh Viên", "Xem Điểm Danh"], key='page')

# Handle navigation from button click
if st.session_state['navigate_to']:
    page = st.session_state['navigate_to']
    st.session_state['navigate_to'] = None  # Reset navigation

if page == "Đăng Ký Sinh Viên":
    st.header("Đăng Ký Sinh Viên Mới")
    
    sessions = get_sessions_list()
    if not sessions:
        st.warning("Chưa có khối thực tập nào. Vui lòng tạo khối thực tập trước.")
    else:
        session_options = [f"{s['class_name']} - {s['session_date']} ({s['session_day']})" for s in sessions]
        selected_session = st.selectbox("Chọn Khối Thực Tập", session_options)
        session_id = sessions[session_options.index(selected_session)]['id']
        
        # Thêm tùy chọn cho phương thức đăng ký ảnh
        registration_method = st.radio("Chọn phương thức đăng ký ảnh", ["Chụp ảnh từ camera", "Tải lên ảnh từ máy tính"])
        
        if registration_method == "Chụp ảnh từ camera":
            image_file = st.camera_input("Chụp ảnh sinh viên")
        else:
            image_file = st.file_uploader("Tải ảnh và nhập thông tin thủ công", type=["jpg", "png", "jpeg"])
            uploaded_files = st.file_uploader("Tải ảnh và đăng ký tự động (Định dạng file: ID_HoTen)", type=["jpg", "png", "jpeg"],
                                              accept_multiple_files=True)
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Ảnh: {uploaded_file.name}", use_container_width=True)
                    file_name = uploaded_file.name
                    if '_' in file_name:
                        parts = file_name.split('_')
                        if len(parts) >= 2:
                            student_id = parts[0]
                            name = parts[1].split('.')[0]
                            st.write(f"Tự động điền: MSSV = {student_id}, Tên = {name}")
                        else:
                            student_id = st.text_input("MSSV", key=f"mssv_{uploaded_file.name}")
                            name = st.text_input("Tên Sinh Viên", key=f"name_{uploaded_file.name}")
                    else:
                        student_id = st.text_input("MSSV", key=f"mssv_{uploaded_file.name}")
                        name = st.text_input("Tên Sinh Viên", key=f"name_{uploaded_file.name}")

                    if name and student_id:  # Tự động đăng ký nếu thông tin đầy đủ
                        existing_name = get_student_name(student_id)
                        if existing_name and existing_name.lower().strip() != name.lower().strip():
                            st.error(f"MSSV {student_id} đã tồn tại với tên '{existing_name}'. Vui lòng nhập đúng tên.")
                        else:
                            img_array = np.array(image)
                            embedding = recognizer.get_embedding(img_array)
                            if embedding is not None:
                                if not os.path.exists('student_images'):
                                    os.makedirs('student_images')
                                image_path = f"student_images/{student_id}_{name}_{datetime.now(tz).strftime('%Y%m%d%H%M%S')}.jpg"
                                image.save(image_path)
                                conn = sqlite3.connect('attendance.db')
                                c = conn.cursor()
                                c.execute(
                                    "INSERT INTO students (id, name, embedding, image_path, session_id) VALUES (?, ?, ?, ?, ?)",
                                    (student_id, name, embedding.tobytes(), image_path, session_id))
                                conn.commit()
                                conn.close()
                                st.success(
                                    f"Đã đăng ký hình ảnh cho sinh viên {name} với MSSV {student_id} thành công!")
                            else:
                                st.error(
                                    f"Không phát hiện khuôn mặt hoặc có nhiều khuôn mặt trong ảnh {uploaded_file.name}. Vui lòng chọn ảnh khác với chỉ một khuôn mặt.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if image_file is not None:
                image = Image.open(image_file)
                st.image(image, caption="Ảnh đã chọn", use_container_width=True)
        with col2:
            excel_file = st.file_uploader("Upload file Excel danh sách sinh viên", type=["xlsx", "xls"])
            if excel_file is not None:
                df = pd.read_excel(excel_file)
                st.write("Danh sách sinh viên từ file Excel:")
                st.dataframe(df)
                student_options = df['Họ tên SV'].tolist()
                selected_student = st.selectbox("Chọn sinh viên để đăng ký", student_options)
                student_id = str(df[df['Họ tên SV'] == selected_student]['MSSV'].values[0])
                name = selected_student
            else:
                name = st.text_input("Tên Sinh Viên")
                student_id = st.text_input("MSSV")
            
            if st.button("Đăng Ký") and image_file is not None and name and student_id:
                existing_name = get_student_name(student_id)
                if existing_name and existing_name.lower().strip() != name.lower().strip():
                    st.error(f"MSSV {student_id} đã tồn tại với tên '{existing_name}'. Vui lòng nhập đúng tên.")
                else:
                    image = Image.open(image_file)
                    img_array = np.array(image)
                    embedding = recognizer.get_embedding(img_array)
                    if embedding is not None:
                        if not os.path.exists('student_images'):
                            os.makedirs('student_images')
                        image_path = f"student_images/{student_id}_{name}_{datetime.now(tz).strftime('%Y%m%d%H%M%S')}.jpg"
                        image.save(image_path)
                        conn = sqlite3.connect('attendance.db')
                        c = conn.cursor()
                        c.execute("INSERT INTO students (id, name, embedding, image_path, session_id) VALUES (?, ?, ?, ?, ?)",
                                  (student_id, name, embedding.tobytes(), image_path, session_id))
                        conn.commit()
                        conn.close()
                        st.success(f"Đã đăng ký hình ảnh cho sinh viên {name} với MSSV {student_id} thành công!")
                    else:
                        st.error("Không phát hiện khuôn mặt hoặc có nhiều khuôn mặt. Vui lòng chọn ảnh khác với chỉ một khuôn mặt.")

elif page == "Tạo Buổi Thực Tập":
    st.header("Tạo Buổi Thực Tập Mới")
    today = datetime.now(tz).date()
    today_str = today.strftime("%Y-%m-%d")
    
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now(tz).time()
    if 'end_time' not in st.session_state:
        st.session_state.end_time = (datetime.now(tz) + timedelta(hours=1)).time()
    
    class_name = st.text_input("Khối lớp thực tập (ví dụ: RHM...)", "")
    session_date = st.date_input("Ngày thực tập", value=today)
    session_day_en = session_date.strftime("%A")
    session_day_vn = get_vietnamese_day(session_day_en)
    st.write(f"Thứ trong tuần: {session_day_vn}")
    start_time = st.time_input("Giờ bắt đầu đánh giá điểm chuyên cần", value=st.session_state.start_time)
    end_time = st.time_input("Giờ kết thúc đánh giá điểm chuyên cần", value=st.session_state.end_time)
    max_attendance_score = st.number_input("Điểm chuyên cần tối đa (1-10)", min_value=1, max_value=10, value=10)
    
    st.session_state.start_time = start_time
    st.session_state.end_time = end_time
    
    if st.button("Tạo Buổi Thực Tập"):
        session_id = create_new_session(class_name, session_date.strftime("%Y-%m-%d"), session_day_vn, start_time.strftime("%H:%M"), end_time.strftime("%H:%M"), max_attendance_score)
        if session_id:
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
        
        record_ids, ids, names, embeddings = load_embeddings_by_session(session_id)
        
        attendance_method = st.selectbox("Chọn phương thức điểm danh", ["Chụp ảnh", "Tải lên ảnh", "Real-time camera"])
        
        if attendance_method == "Chụp ảnh":
            image_file = st.camera_input("Chụp ảnh để điểm danh")
            if image_file is not None:
                image = Image.open(image_file)
                img_array = np.array(image)
                faces = recognizer.app.get(img_array)
                if len(faces) == 1:
                    embedding = faces[0].embedding
                    record_id, student_id, student_name = find_closest_match(embedding, record_ids, ids, names, embeddings)
                    if record_id is not None:
                        if not check_attendance(session_id, student_id):
                            now = datetime.now(tz)
                            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                            start_time = datetime.strptime(f"{session_info['session_date']} {session_info['start_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                            end_time = datetime.strptime(f"{session_info['session_date']} {session_info['end_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                            late_threshold = end_time + timedelta(minutes=15)
                            
                            if start_time <= now <= end_time:
                                attendance_score = session_info['max_attendance_score']
                                note = ""
                            elif end_time < now < late_threshold:
                                attendance_score = 0
                                note = ""
                            elif late_threshold <= now:
                                attendance_score = 0
                                note = "Trễ >15p"
                            else:
                                attendance_score = 0
                                note = "Điểm danh sau giờ kết thúc"
                            
                            mark_attendance(session_id, student_id, timestamp, attendance_score, note)
                            message = f"Đã điểm danh: {student_name} (MSSV: {student_id}) lúc {timestamp} - Điểm chuyên cần: {attendance_score}"
                            if note:
                                message += f" - {note}"
                            st.success(message)
                            
                            # Hiển thị hình ảnh của sinh viên
                            image_path = get_student_image(record_id)
                            if image_path and os.path.exists(image_path):
                                student_image = Image.open(image_path)
                                st.image(student_image.resize((300, 300)), caption=f"Hình ảnh của {student_name} (MSSV: {student_id})")
                        else:
                            st.warning(f"Sinh viên {student_name} (MSSV: {student_id}) đã được điểm danh trong buổi thực tập này.")
                    else:
                        st.error("Không nhận diện được sinh viên trong ảnh.")
                else:
                    st.error("Ảnh không chứa đúng một khuôn mặt. Vui lòng chụp lại.")
        
        elif attendance_method == "Tải lên ảnh":
            uploaded_file = st.file_uploader("Tải lên ảnh để điểm danh", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                faces = recognizer.app.get(img_array)
                if len(faces) == 1:
                    embedding = faces[0].embedding
                    record_id, student_id, student_name = find_closest_match(embedding, record_ids, ids, names, embeddings)
                    if record_id is not None:
                        if not check_attendance(session_id, student_id):
                            now = datetime.now(tz)
                            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                            start_time = datetime.strptime(f"{session_info['session_date']} {session_info['start_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                            end_time = datetime.strptime(f"{session_info['session_date']} {session_info['end_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                            late_threshold = end_time + timedelta(minutes=15)
                            
                            if start_time <= now <= end_time:
                                attendance_score = session_info['max_attendance_score']
                                note = ""
                            elif end_time < now < late_threshold:
                                attendance_score = 0
                                note = ""
                            elif late_threshold <= now:
                                attendance_score = 0
                                note = "Trễ >15p"
                            else:
                                attendance_score = 0
                                note = "Điểm danh sau giờ kết thúc"
                            
                            mark_attendance(session_id, student_id, timestamp, attendance_score, note)
                            message = f"Đã điểm danh: {student_name} (MSSV: {student_id}) lúc {timestamp} - Điểm chuyên cần: {attendance_score}"
                            if note:
                                message += f" - {note}"
                            st.success(message)
                            
                            # Hiển thị hình ảnh của sinh viên
                            image_path = get_student_image(record_id)
                            if image_path and os.path.exists(image_path):
                                student_image = Image.open(image_path)
                                st.image(student_image.resize((300, 300)), caption=f"Hình ảnh của {student_name} (MSSV: {student_id})")
                        else:
                            st.warning(f"Sinh viên {student_name} (MSSV: {student_id}) đã được điểm danh trong buổi thực tập này.")
                    else:
                        st.error("Không nhận diện được sinh viên trong ảnh.")
                else:
                    st.error("Ảnh không chứa đúng một khuôn mặt. Vui lòng tải lên ảnh khác.")
        
        elif attendance_method == "Real-time camera":
            st.write("Chế độ điểm danh tự động...")
    
            # Khởi tạo trạng thái trong st.session_state nếu chưa có
            #if 'last_attended_student' not in st.session_state:
            #    st.session_state['last_attended_student'] =None
    
            image = camera_input_live()
            if image is not None:
                image = Image.open(image)
                img_array = np.array(image)
                if img_array is not None:
                # Chuyển đổi từ 4 kênh sang 3 kênh nếu cần
                    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]  # Loại bỏ kênh alpha
                    faces = recognizer.app.get(img_array)
                    if len(faces) == 1:
                        embedding = faces[0].embedding
                        record_id, student_id, student_name = find_closest_match(embedding, record_ids, ids, names, embeddings)
                        if record_id is not None:

                            image_path = get_student_image(record_id)
                            student_image = Image.open(image_path)

                            if not check_attendance(session_id, student_id):
                                now = datetime.now(tz)
                                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                start_time = datetime.strptime(f"{session_info['session_date']} {session_info['start_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                                end_time = datetime.strptime(f"{session_info['session_date']} {session_info['end_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                                late_threshold = end_time + timedelta(minutes=15)
                
                                if start_time <= now <= end_time:
                                    attendance_score = session_info['max_attendance_score']
                                    note = ""
                                elif end_time < now < late_threshold:
                                    attendance_score = 0
                                    note = ""
                                elif late_threshold <= now:
                                    attendance_score = 0
                                    note = "Trễ >15p"
                                else:
                                    attendance_score = 0
                                    note = "Điểm danh sau giờ kết thúc"
                
                                mark_attendance(session_id, student_id, timestamp, attendance_score, note)
                                message = f"Đã điểm danh: {student_name} (MSSV: {student_id}) lúc {timestamp} - Điểm chuyên cần: {attendance_score}"
                                if note:
                                    message += f" - {note}"
                                st.success(message)
                
                                # Cập nhật thông tin và hình ảnh của sinh viên vào st.session_state
                                #image_path = get_student_image(record_id)
                                #if image_path and os.path.exists(image_path):
                                #    student_image = Image.open(image_path)
                                    #st.session_state['last_attended_student'] = {
                                    #'name': student_name,
                                    #'id': student_id,
                                    #'image': student_image
                                    #}

                                #    st.subheader(
                                #        f"Sinh viên đã được điểm danh: {student_name} (MSSV: {student_id})")

                                    #st.rerun()
                            else:
                                #pass
                                st.warning(f"Sinh viên {student_name} (MSSV: {student_id}) đã được điểm danh trong buổi thực tập này.")
                                st.image(student_image.resize((300, 300)),
                                         caption=f"Hình ảnh của {student_name} (MSSV: {student_id})")
                        else:
                            pass
                            #st.error("Không nhận diện được sinh viên trong ảnh.")
                    else:
                        pass
                        #st.error("Ảnh không chứa đúng một khuôn mặt. Vui lòng thử lại.")
    
            # Hiển thị thông tin và hình ảnh của sinh viên đã điểm danh gần nhất
            #if st.session_state['last_attended_student'] is not None:
            #    student = st.session_state['last_attended_student']
            #    st.subheader(f"Sinh viên đã điểm danh gần nhất: {student['name']} (MSSV: {student['id']})")
            #    st.image(student['image'].resize((300, 300)), caption=f"Hình ảnh của {student['name']} (MSSV: {student['id']})")

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
            df = pd.DataFrame(attendance_list, columns=['MSSV', 'Họ tên SV', 'Giờ điểm danh', 'Điểm', 'Ghi chú', 'Khối thực tập', 'Ngày', 'Thứ', 'Giờ bắt đầu', 'Giờ kết thúc'])
            st.dataframe(df)
            
            if st.button("Tải về Danh Sách Điểm Danh (Excel)"):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Điểm Danh')
                excel_data = output.getvalue()
                st.download_button(
                    label="Tải về file Excel",
                    data=excel_data,
                    file_name="danh_sach_diem_danh.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            st.subheader("Xóa Record Điểm Danh")
            selected_student_id = st.selectbox("Chọn MSSV để xóa", df['MSSV'])
            if st.button("Xóa Record Này"):
                conn = get_db_connection()
                conn.execute("DELETE FROM attendance WHERE session_id = ? AND student_id = ?", (session_id, selected_student_id))
                conn.commit()
                conn.close()
                st.success(f"Đã xóa record điểm danh của sinh viên {selected_student_id}.")
                st.rerun()
        else:
            st.write("Không có sinh viên nào được ghi nhận.")
