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
import pandas as pd
from io import BytesIO

# Thiết lập múi giờ Việt Nam (UTC+7)
tz = pytz.timezone('Asia/Ho_Chi_Minh')

# Khởi tạo cơ sở dữ liệu
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id TEXT PRIMARY KEY, name TEXT, embedding BLOB, image_path TEXT, session_id INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY, class_name TEXT, session_date TEXT, session_day TEXT, start_time TEXT, end_time TEXT, max_attendance_score INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (session_id INTEGER, student_id TEXT, status TEXT, timestamp TEXT, attendance_score INTEGER)''')
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
    students = conn.execute('SELECT id, name, image_path, session_id FROM students WHERE session_id = ?', (session_id,)).fetchall()
    conn.close()
    return students

# Lấy danh sách khối thực tập
def get_sessions_list():
    conn = get_db_connection()
    sessions = conn.execute('SELECT id, class_name, session_date, session_day FROM sessions').fetchall()
    conn.close()
    return sessions

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
    
    df = pd.DataFrame(students, columns=['id', 'name', 'image_path', 'session_id'])
    st.dataframe(df[['id', 'name']])
    
    selected_student_id = st.selectbox("Chọn sinh viên để xem hình ảnh", df['id'])
    selected_student = df[df['id'] == selected_student_id].iloc[0]
    
    image_path = selected_student['image_path']
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=f"Hình ảnh của {selected_student['name']}", use_column_width=True)
    else:
        st.warning(f"Không tìm thấy hình ảnh cho sinh viên {selected_student['name']}.")
    
    if st.button("Xóa Sinh Viên Này"):
        conn = get_db_connection()
        conn.execute("DELETE FROM students WHERE id = ?", (selected_student_id,))
        conn.commit()
        conn.close()
        st.success(f"Đã xóa sinh viên {selected_student['name']}.")
        st.rerun()
    
    if st.button("Tải về Danh Sách Sinh Viên (Excel)"):
        df_to_export = df[['id', 'name']]
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
    c.execute("SELECT id, name, embedding FROM students WHERE session_id = ?", (session_id,))
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
        SELECT s.id, s.name, a.timestamp, a.attendance_score, ses.class_name, ses.session_date, ses.session_day, ses.start_time, ses.end_time
        FROM students s
        JOIN attendance a ON s.id = a.student_id
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
        
        col1, col2 = st.columns([2, 1])
        with col1:
            image_file = st.camera_input("Chụp ảnh sinh viên")
        with col2:
            excel_file = st.file_uploader("Upload file Excel danh sách sinh viên", type=["xlsx", "xls"])
            if excel_file is not None:
                df = pd.read_excel(excel_file)
                st.write("Danh sách sinh viên từ file Excel:")
                st.dataframe(df)
                student_options = df['Họ tên SV'].tolist()
                selected_student = st.selectbox("Chọn sinh viên để đăng ký", student_options)
                student_id = str(df[df['Họ tên SV'] == selected_student]['MSSV'].values[0])  # Chuyển thành chuỗi để đồng nhất kiểu dữ liệu
                name = selected_student
            else:
                name = st.text_input("Tên Sinh Viên")
                student_id = None  # Sẽ được gán sau
            
            if st.button("Đăng Ký") and image_file is not None and name:
                image = Image.open(image_file)
                img_array = np.array(image)
                embedding = recognizer.get_embedding(img_array)
                if embedding is not None:
                    if not os.path.exists('student_images'):
                        os.makedirs('student_images')
                    image_path = f"student_images/{name}_{datetime.now(tz).strftime('%Y%m%d%H%M%S')}.jpg"
                    image.save(image_path)
                    conn = sqlite3.connect('attendance.db')
                    c = conn.cursor()
                    if student_id is None:
                        # Tạo ID mới nếu không có file Excel
                        c.execute("SELECT MAX(CAST(id AS INTEGER)) FROM students WHERE id GLOB '[0-9]*'")
                        max_id = c.fetchone()[0]
                        student_id = str(max_id + 1) if max_id is not None else "1"
                    else:
                        # Kiểm tra xem MSSV đã tồn tại chưa
                        c.execute("SELECT id FROM students WHERE id = ?", (student_id,))
                        if c.fetchone() is not None:
                            st.error(f"MSSV {student_id} đã tồn tại. Vui lòng kiểm tra lại file Excel.")
                            conn.close()
                    # Chèn dữ liệu vào cơ sở dữ liệu với các cột được phân tách rõ ràng
                    c.execute("INSERT INTO students (id, name, embedding, image_path, session_id) VALUES (?, ?, ?, ?, ?)",
                              (student_id, name, embedding.tobytes(), image_path, session_id))
                    conn.commit()
                    conn.close()
                    st.success(f"Đã đăng ký sinh viên {name} với MSSV {student_id} thành công!")
                else:
                    st.error("Không phát hiện khuôn mặt hoặc có nhiều khuôn mặt. Vui lòng chụp lại với chỉ một khuôn mặt.")

elif page == "Tạo Buổi Thực Tập":
    st.header("Tạo Buổi Thực Tập Mới")
    today = datetime.now(tz).date()
    today_str = today.strftime("%Y-%m-%d")
    
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now(tz).time()
    if 'end_time' not in st.session_state:
        st.session_state.end_time = (datetime.now(tz) + timedelta(hours=1)).time()
    
    class_name = st.text_input("Khối lớp thực tập (ví dụ: Lớp 10A)", "")
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
        
        ids, names, embeddings = load_embeddings_by_session(session_id)
        
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
                    now = datetime.now(tz)
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    start_time = datetime.strptime(f"{session_info['session_date']} {session_info['start_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                    end_time = datetime.strptime(f"{session_info['session_date']} {session_info['end_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
                    attendance_score = session_info['max_attendance_score'] if start_time <= now <= end_time else 0
                    mark_attendance(session_id, student_id, timestamp, attendance_score)
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
            df = pd.DataFrame(attendance_list, columns=['MSSV', 'Họ tên SV', 'Giờ điểm danh', 'Điểm', 'Khối thực tập', 'Ngày', 'Thứ', 'Giờ bắt đầu', 'Giờ kết thúc'])
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
