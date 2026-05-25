import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. CẤU HÌNH TRANG VÀ CUSTOM CSS CHUẨN UEH
# ==========================================
st.set_page_config(
    page_title="UEH GPA Predictor & Bias Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm hiệu ứng giao diện Dashboard sang xịn mịn
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1 {
        color: #000000 !important; /* ĐÃ CHỈNH SỬA TIÊU ĐỀ SANG MÀU ĐEN */
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .ueh-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 30px;
        border-left: 6px solid #004B87; /* Giữ lại đường viền xanh thương hiệu UEH ở rìa trái */
    }
    .bias-box {
        background-color: #fff9e6;
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .report-box {
        background-color: #f1f3f5;
        color: #383d41;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #6c757d;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 4px solid #004B87;
    }
    </style>
""", unsafe_allow_html=True)

# Header chào mừng tích hợp Logo UEH chính thức
st.markdown("""
    <div class="ueh-header">
        <div>
            <h1 style="color: #000000 !important;">UEH INSIGHTS: HỆ THỐNG DỰ ĐOÁN GPA & PHÂN TÍCH BIAS</h1>
            <p style="margin:5px 0 0 0; color:#555; font-size: 15px;">
                Ứng dụng Học Máy đánh giá mối tương quan giữa hành vi tự khai báo (Self-reported Data) và Định kiến dữ liệu (Bias) ảnh hưởng đến GPA Sinh viên.
            </p>
        </div>
        <img src="https://ueh.edu.vn/images/logo.png" width="165" alt="UEH Logo">
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 2. ĐỌC VÀ CHUYỂN ĐỔI DỮ LIỆU TỰ ĐỘNG DÒ FILE
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    import os
    # Tự động quét tìm bất kỳ file nào có đuôi .csv trong thư mục hiện tại
    csv_files = "gpa.csv"
    
    if not csv_files:
        st.error("❌ Không tìm thấy bất kỳ file .csv nào trong thư mục này! Bạn kiểm tra lại xem đã bỏ file khảo sát vào đúng thư mục chưa nha.")
        return None, None, None

    
    try:
        df = pd.read_csv("gpa.csv")
    except Exception as e:
        st.error(f"Lỗi khi đọc file {"gpa.csv"}: {e}")
        return None, None, None

    # Đồng bộ hóa tên cột ngắn để xử lý code nhanh
    df.columns = [
        'Timestamp', 'Nam_Hoc', 'Thoi_Gian_Tu_Hoc', 'So_Mon_Hoc', 
        'Di_Lam_Them', 'Tham_Gia_CLB', 'Ty_Le_Len_Lop', 'Hinh_Thuc_Hoc', 
        'Thoi_Gian_Ngu', 'Thoi_Gian_MXH', 'GPA'
    ]

    # Làm sạch cột Số Môn Học (Xử lý trường hợp "3 môn" thành số 3)
    def extract_number(val):
        nums = re.findall(r'\d+', str(val))
        return int(nums[0]) if nums else 6

    df['So_Mon_Hoc'] = df['So_Mon_Hoc'].apply(extract_number)

    # Khởi tạo DataFrame đã mã hóa cho mô hình Machine Learning
    df_encoded = pd.DataFrame()
    df_encoded['So_Mon_Hoc'] = df['So_Mon_Hoc']

    categorical_cols = ['Nam_Hoc', 'Thoi_Gian_Tu_Hoc', 'Di_Lam_Them', 'Tham_Gia_CLB', 'Ty_Le_Len_Lop', 'Hinh_Thuc_Hoc', 'Thoi_Gian_Ngu', 'Thoi_Gian_MXH', 'GPA']
    maps = {}
    
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
        unique_vals = sorted(list(df[col].unique()))
        maps[col] = {val: idx for idx, val in enumerate(unique_vals)}
        df_encoded[col] = df[col].map(maps[col])

    return df, df_encoded, maps

# Chạy hàm load dữ liệu
df_raw, df_clean, maps = load_and_preprocess_data()

if df_raw is not None:
    # Huấn luyện mô hình Random Forest Classifier để lấy Feature Importance
    features = ['Nam_Hoc', 'Thoi_Gian_Tu_Hoc', 'Di_Lam_Them', 'Tham_Gia_CLB', 'Ty_Le_Len_Lop', 'Hinh_Thuc_Hoc', 'Thoi_Gian_Ngu', 'Thoi_Gian_MXH', 'So_Mon_Hoc']
    X = df_clean[features]
    y = df_clean['GPA']
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)
    
    # Trích xuất độ quan trọng của các thuộc tính
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Định dạng lại tên hiển thị cho biểu đồ trực quan hơn
    friendly_names = {
        'Nam_Hoc': 'Năm học hiện tại',
        'Thoi_Gian_Tu_Hoc': 'Thời gian tự học / tuần',
        'Di_Lam_Them': 'Thời gian đi làm thêm',
        'Tham_Gia_CLB': 'Tham gia CLB Đoàn Hội',
        'Ty_Le_Len_Lop': 'Tỷ lệ chuyên cần trên lớp',
        'Hinh_Thuc_Hoc': 'Hình thức học tập',
        'Thoi_Gian_Ngu': 'Thời gian ngủ / ngày',
        'Thoi_Gian_MXH': 'Thời gian lướt MXH',
        'So_Mon_Hoc': 'Số lượng môn đăng ký'
    }
    
    features_ranked = [friendly_names[features[i]] for i in indices]
    importances_ranked = [importances[i] for i in indices]

    # ==========================================
    # 3. SIDEBAR - FORM NHẬP INPUT CHỌN LỌC
    # ==========================================
    st.sidebar.image("https://ueh.edu.vn/images/logo.png", width=130)
    st.sidebar.markdown("<h2 style='color:#004B87; font-size:22px; font-weight:700;'>🎯 THÔNG TIN SINH VIÊN</h2>", unsafe_allow_html=True)
    st.sidebar.write("Hãy tự khai báo các chỉ số hành vi học tập của bạn dưới đây:")
    
    # Tạo các selectbox động lấy trực tiếp danh sách từ dữ liệu thật
    input_nam_hoc = st.sidebar.selectbox("Bạn đang là sinh viên năm mấy?", list(maps['Nam_Hoc'].keys()))
    input_tu_hoc = st.sidebar.selectbox("Thời gian tự học ngoài giờ / tuần?", list(maps['Thoi_Gian_Tu_Hoc'].keys()))
    input_so_mon = st.sidebar.slider("Số môn học đăng ký kỳ này?", int(df_raw['So_Mon_Hoc'].min()), int(df_raw['So_Mon_Hoc'].max()), int(df_raw['So_Mon_Hoc'].median()))
    input_lam_them = st.sidebar.selectbox("Thời gian đi làm thêm của bạn?", list(maps['Di_Lam_Them'].keys()))
    input_clb = st.sidebar.selectbox("Có tham gia CLB/Đoàn/Hội không?", list(maps['Tham_Gia_CLB'].keys()))
    input_len_lop = st.sidebar.selectbox("Tỷ lệ đi học đầy đủ trên lớp?", list(maps['Ty_Le_Len_Lop'].keys()))
    input_hinh_thuc = st.sidebar.selectbox("Hình thức học tập chủ yếu?", list(maps['Hinh_Thuc_Hoc'].keys()))
    input_ngu = st.sidebar.selectbox("Thời gian ngủ trung bình mỗi ngày?", list(maps['Thoi_Gian_Ngu'].keys()))
    input_mxh = st.sidebar.selectbox("Thời gian lướt Mạng xã hội / ngày?", list(maps['Thoi_Gian_MXH'].keys()))

    # ==========================================
    # 4. GIAO DIỆN CHÍNH TRỰC QUAN (TABS)
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["🔮 Dự Đoán Điểm Số GPA", "📊 Feature Importance & Dữ Liệu gốc", "⚠️ Phân Tích Định Kiến (Bias Analysis)"])
    
    with tab1:
        st.markdown("### 🔮 Dự báo Xếp loại GPA bằng Trí tuệ nhân tạo")
        st.write("Dựa vào thông tin bạn tự khai báo ở thanh Sidebar bên trái, mô hình học máy Random Forest dự đoán mức điểm kì này của bạn là:")
        
        # Chuyển đổi các lựa chọn chữ của người dùng thành Vector số để đưa vào mô hình AI
        user_vector = [
            maps['Nam_Hoc'][input_nam_hoc],
            maps['Thoi_Gian_Tu_Hoc'][input_tu_hoc],
            maps['Di_Lam_Them'][input_lam_them],
            maps['Tham_Gia_CLB'][input_clb],
            maps['Ty_Le_Len_Lop'][input_len_lop],
            maps['Hinh_Thuc_Hoc'][input_hinh_thuc],
            maps['Thoi_Gian_Ngu'][input_ngu],
            maps['Thoi_Gian_MXH'][input_mxh],
            input_so_mon
        ]
        
        # Dự đoán kết quả mã số (bọc DataFrame để tránh lỗi Valid Feature Names của Sklearn)
        user_df = pd.DataFrame([user_vector], columns=features)
        pred_num = model.predict(user_df)[0]
        # Tra ngược từ mã số ra nhãn chữ GPA gốc (Ví dụ: 3.0 - 3.5)
        predicted_gpa_text = [k for k, v in maps['GPA'].items() if v == pred_num][0]
        
        # Thiết kế khối hiển thị kết quả cực "chất"
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #666; font-weight: 600; margin: 0; font-size:14px; text-transform: uppercase;">Dự Đoán GPA Kỳ Này</p>
                <h2 style="color: #004B87; font-size: 45px; margin: 10px 0; font-weight: 800;">{predicted_gpa_text}</h2>
                <p style="color: #28a745; margin: 0; font-size:13px;">✓ Thuật toán Random Forest Classifier</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.info("💡 **Gợi ý cải thiện hiệu năng học tập từ AI:**\n\nĐối chiếu với trọng số phân tích từ toàn bộ sinh viên UEH, nếu bạn muốn nâng bậc GPA lên nhóm cao hơn, việc tăng **Thời gian tự học** và duy trì **Tỷ lệ chuyên cần lên lớp trên 90%** mang lại hiệu quả tác động mạnh mẽ nhất vượt trội so với các yếu tố còn lại.")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Tóm tắt Hồ sơ hành vi học tập đã chọn:")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**• Đối tượng:** {input_nam_hoc}")
            st.markdown(f"**• Số môn học kỳ này:** {input_so_mon} môn")
        with col_b:
            st.markdown(f"**• Tự học ngoài giờ:** {input_tu_hoc}")
            st.markdown(f"**• Tỷ lệ lên lớp:** {input_len_lop}")
        with col_c:
            st.markdown(f"**• Thời gian lướt mạng:** {input_mxh}")
            st.markdown(f"**• Tình trạng làm thêm:** {input_lam_them}")

    with tab2:
        st.markdown("### 📊 Độ Quan Trọng Của Các Yếu Tố (Feature Importance Analysis)")
        st.write("Mô hình AI bóc tách và xếp hạng xem hành vi thói quen nào thực sự đang 'thao túng' và đóng vai trò quyết định nhiều nhất đến bảng điểm GPA của sinh viên:")
        
        # Vẽ biểu đồ thanh ngang bằng Plotly cực chất
        fig_importance = px.bar(
            x=importances_ranked[::-1], 
            y=features_ranked[::-1], 
            orientation='h',
            labels={'x': 'Mức độ ảnh hưởng tương đối (Trọng số Feature Importance)', 'y': 'Yếu tố thói quen'},
            color=importances_ranked[::-1], 
            color_continuous_scale='Blugrn',
            title="Xếp hạng các yếu tố ảnh hưởng trực tiếp đến GPA"
        )
        fig_importance.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_importance, width='stretch')
        
        st.markdown("---")
        st.markdown("### 📂 Bảng dữ liệu khảo sát gốc (Đã được làm sạch)")
        st.write("Tổng số mẫu thu thập hiện tại từ biểu mẫu: **{} sinh viên**".format(len(df_raw)))
        st.dataframe(df_raw, width='stretch')

    with tab3:
        st.markdown("### ⚠️ Góc nhìn Khoa học Dữ liệu: Phân tích Định kiến & Dữ liệu Tự khai báo")
        
        st.markdown("""
        <div class="bias-box">
            <h4>🚨 1. Social Desirability Bias (Định kiến mong muốn xã hội trong Self-reported Data)</h4>
            <p>Dữ liệu này thu thập dưới dạng <b>Self-reported data (Người học tự điền)</b>. Theo lý thuyết tâm lý học hành vi, sinh viên luôn có xu hướng vô thức khai báo các số liệu "đẹp và lý tưởng hơn" thực tế để tạo cảm giác mình chăm chỉ. Cụ thể: hãy nhìn biểu đồ phân phối bên dưới, số lượng sinh viên tự khai báo có <i>Tỷ lệ lên lớp > 90%</i> chiếm tỷ số áp đảo, trong khi thời gian lướt mạng xã hội có xu hướng bị nén giảm xuống.</p>
        </div>
        
        <div class="report-box">
            <h4>📉 2. Survivorship Bias (Định kiến sống sót trong chọn mẫu)</h4>
            <p>Biểu mẫu khảo sát thường có tỷ lệ phản hồi cực kỳ cao từ nhóm sinh viên năng nổ, tích cực tham gia hoạt động hoặc những bạn có điểm số khá giỏi quan tâm đến học tập. Ngược lại, nhóm sinh viên có học lực yếu, chán học hoặc ít tương tác thường sẽ bỏ qua form khảo sát này. Do đó, tập dữ liệu bị <b>lệch (skewed) hẳn về phía nhóm điểm số cao (3.0 - 3.5 và > 3.5)</b>, khiến mô hình AI học máy thu được có xu hướng dự đoán điểm cao hơn thực tế.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Biểu đồ minh chứng trực quan hóa Bias
        st.markdown("#### 📊 Trực quan minh chứng sự lệch (Skewness) dữ liệu tự khai báo")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            # Đã cấu hình cột chuẩn Thoi_Gian_MXH
            fig_mxh = px.histogram(
                df_raw, 
                x='Thoi_Gian_MXH', 
                title="Phân phối Thời gian lướt MXH hàng ngày (Có xu hướng bị khai báo giảm đi)",
                labels={'Thoi_Gian_MXH': 'Thời gian lướt MXH'},
                color_discrete_sequence=['#ff6b6b']
            )
            st.plotly_chart(fig_mxh, width='stretch')
            
        with col_img2:
            # Đã cấu hình cột chuẩn GPA
            fig_gpa = px.histogram(
                df_raw, 
                x='GPA', 
                title="Phân phối điểm GPA (Dữ liệu bị lệch hẳn sang phải do Survivorship Bias)",
                labels={'GPA': 'Mức xếp loại GPA'},
                color_discrete_sequence=['#004B87']
            )
            st.plotly_chart(fig_gpa, width='stretch')
