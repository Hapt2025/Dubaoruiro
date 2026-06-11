import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------
# STEP 1: PAGE CONFIGURATION (Lệnh Streamlit đầu tiên)
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# -----------------------------------------------------------------------------
# STEP 2: CACHED DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để tránh rerun và hỗ trợ cache."""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_bytes)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# Khởi tạo danh sách biến đặc trưng cố định theo Notebook
FEATURES = [f"X_{i}" for i in range(1, 15)]
TARGET = "default"

# -----------------------------------------------------------------------------
# STEP 3: SIDEBAR - VÙNG CẤU HÌNH
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Upload dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (CSV/XLSX)", 
        type=["csv", "xlsx"],
        help="Chọn file dataset1.csv hoặc định dạng tương đương có các cột X_1 đến X_14 và default."
    )
    
    st.divider()
    
    # Lựa chọn mô hình (Notebook chứa 3 mô hình)
    st.subheader("🤖 Lựa chọn Mô hình")
    model_choice = st.selectbox(
        "Chọn thuật toán huấn luyện:",
        options=["Logistic Regression", "Decision Tree", "Random Forest"],
        index=2, # Mặc định chọn Random Forest như bước cuối của notebook
        help="Chọn thuật toán để xây dựng mô hình phân loại giao dịch."
    )
    
    # Tham số mô hình động
    st.subheader("🎛️ Siêu tham số AI")
    
    # Cấu hình tham số chung và riêng theo từng mô hình dựa trên notebook
    params = {}
    params['random_state'] = st.number_input("Random State", value=42, step=1, help="Đảm bảo tính tái lập kết quả.")
    params['test_size'] = st.slider("Tỷ lệ tập kiểm tra (Test Size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    
    with st.expander("Tham số nâng cao thuật toán"):
        if model_choice == "Logistic Regression":
            params['max_iter'] = st.number_input("Max Iterations", value=1000, step=100)
            params['C'] = st.slider("Inverse Regularization (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        
        elif model_choice == "Decision Tree":
            params['criterion'] = st.selectbox("Criterion", options=["gini", "entropy", "log_loss"], index=0)
            params['max_depth'] = st.slider("Max Depth", min_value=1, max_value=50, value=10, step=1)
            
        elif model_choice == "Random Forest":
            params['n_estimators'] = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10)
            params['criterion'] = st.selectbox("Criterion", options=["gini", "entropy", "log_loss"], index=0)
            params['max_depth'] = st.slider("Max Depth", min_value=1, max_value=50, value=15, step=1)

    st.divider()
    
    # Nút kích hoạt huấn luyện duy nhất
    btn_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# STEP 4: HEADER - VÙNG ĐỊNH HƯỚNG
# -----------------------------------------------------------------------------
st.title("🛡️ Hệ thống Phát hiện Giao dịch Gian lận")
st.caption("Ứng dụng hỗ trợ phân tích rủi ro tín dụng và phát hiện hành vi gian lận giao dịch tự động dựa trên học máy.")

if uploaded_file is None:
    st.info("👋 Vui lòng tải tệp dữ liệu mẫu (`.csv` hoặc `.xlsx`) từ Sidebar để bắt đầu.")
    st.stop()

# Đọc dữ liệu khi đã upload
df_main = load_data(uploaded_file, uploaded_file.name)

if df_main is None:
    st.error("Không thể đọc tệp. Vui lòng kiểm tra lại định dạng.")
    st.stop()

# Kiểm tra schema dữ liệu tối thiểu
missing_cols = [col for col in FEATURES + [TARGET] if col not in df_main.columns]
if missing_cols:
    st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc: {missing_cols}")
    st.stop()

st.caption(f"📁 Đang sử dụng tệp: **{uploaded_file.name}**")
st.divider()

# -----------------------------------------------------------------------------
# KHỐI HUẤN LUYỆN (Chỉ chạy khi bấm nút và lưu vào session_state)
# -----------------------------------------------------------------------------
if btn_train:
    with st.spinner("⏳ Đang xử lý dữ liệu và huấn luyện mô hình..."):
        X = df_main[FEATURES]
        y = df_main[TARGET]
        
        # Phân tách tập dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params['test_size'], random_state=params['random_state'], stratify=y
        )
        
        # Tiền xử lý chuẩn hóa bằng StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Khởi tạo thuật toán theo lựa chọn
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=params['max_iter'], C=params['C'], random_state=params['random_state'])
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'], random_state=params['random_state'])
        else: # Random Forest
            model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], max_depth=params['max_depth'], random_state=params['random_state'])
        
        # Fit mô hình
        model.fit(X_train_scaled, y_train)
        
        # Dự đoán kiểm định
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Lưu trữ kết quả vào session_state
        st.session_state['trained_model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['model_name'] = model_choice
        st.session_state['metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist() if y_prob is not None else None
        }
        st.success(f"🎉 Đã huấn luyện thành công mô hình **{model_choice}**!")

# -----------------------------------------------------------------------------
# STEP 5: TABS GIAO DIỆN CHÍNH
# -----------------------------------------------------------------------------
tab_overview, tab_viz, tab_report, tab_inference = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🔬 Kết quả huấn luyện", 
    "🔮 Sử dụng mô hình"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab_overview:
    st.subheader("🔍 Phân tích cấu trúc dữ liệu")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Số lượng bản ghi (Dòng)", f"{df_main.shape[0]:,}")
    col_m2.metric("Số lượng cột đặc trưng", f"{len(FEATURES)}")
    col_m3.metric("Kích thước file", f"{uploaded_file.size / (1024*1024):.2f} MB")
    
    st.write("##### 📑 Xem trước dữ liệu thô (5 dòng đầu):")
    st.dataframe(df_main.head(5), use_container_width=True)
    
    st.write("##### 📈 Thống kê mô tả các biến đặc trưng đưa vào mô hình:")
    st.dataframe(df_main[FEATURES + [TARGET]].describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab_viz:
    st.subheader("🖼️ Phân phối và tương quan đặc trưng")
    
    # Trực quan hóa phân phối lớp mục tiêu trước
    fig_target = px.histogram(
        df_main, x=TARGET, color=TARGET, 
        title="Phân phối của biến mục tiêu (default)",
        labels={'default': 'Trạng thái (0: Hợp lệ, 1: Gian lận)'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.write("##### Biểu đồ phân phối các biến đầu vào tùy chọn:")
    selected_features = st.multiselect(
        "Chọn các biến đặc trưng để hiển thị biểu đồ phân phối (Tối đa nên chọn 4):",
        options=FEATURES,
        default=FEATURES[:4]
    )
    
    if selected_features:
        # Tạo lưới hiển thị linh hoạt
        cols_viz = st.columns(2)
        for idx, feat in enumerate(selected_features):
            col_target = cols_viz[idx % 2]
            with col_target:
                fig_feat = px.box(
                    df_main, x=TARGET, y=feat, color=TARGET,
                    title=f"Phân phối đặc trưng {feat} theo nhãn mục tiêu",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một biến đặc trưng để hiển thị.")

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH ---
with tab_report:
    st.subheader("🔬 Đánh giá hiệu năng mô hình")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Vui lòng bấm nút **[🚀 Huấn luyện mô hình]** ở thanh Sidebar bên trái để xem kết quả kiểm định.")
    else:
        res = st.session_state['metrics']
        st.write(f"⚙️ Mô hình hiện tại trong session: **{st.session_state['model_name']}**")
        
        # Chỉ số vô hướng
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Accuracy (Độ chính xác)", f"{res['accuracy']:.4f}")
        col_r2.metric("Precision (Độ chuẩn xác)", f"{res['precision']:.4f}")
        col_r3.metric("Recall (Độ nhạy)", f"{res['recall']:.4f}")
        col_r4.metric("F1-Score", f"{res['f1']:.4f}")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.write("##### 🔢 Ma trận nhầm lẫn (Confusion Matrix):")
            cm = confusion_matrix(res['y_test'], res['y_pred'])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng"),
                x=['Hợp lệ (0)', 'Gian lận (1)'],
                y=['Hợp lệ (0)', 'Gian lận (1)'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_g2:
            st.write("##### 📋 Báo cáo phân loại chi tiết (Classification Report):")
            report_dict = classification_report(res['y_test'], res['y_pred'], output_dict=True)
            df_report = pd.DataFrame(report_dict).transpose()
            st.dataframe(df_report.style.format(precision=4), use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH ---
with tab_inference:
    st.subheader("🔮 Dự báo rủi ro gian lận giao dịch")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Vui lòng bấm nút **[🚀 Huấn luyện mô hình]** ở thanh Sidebar trước khi thực hiện dự báo dữ liệu mới.")
    else:
        model = st.session_state['trained_model']
        scaler = st.session_state['scaler']
        
        mode = st.radio(
            "Phương thức nhập dữ liệu đầu vào:",
            options=["Nhập trực tiếp từ giao diện", "Tải lên file dữ liệu tổng hợp (X_new)"],
            horizontal=True
        )
        
        if mode == "Nhập trực tiếp từ giao diện":
            st.write("##### Chỉnh sửa thông số giao dịch cụ thể:")
            
            # Khởi tạo form nhập dữ liệu trực tiếp
            with st.form("single_inference_form"):
                inputs = {}
                # Chia biểu mẫu thành 3 cột cho gọn gàng giao diện
                cols_form = st.columns(3)
                
                for idx, feat in enumerate(FEATURES):
                    col_form = cols_form[idx % 3]
                    # Lấy giá trị median, min, max thực tế của tập dữ liệu huấn luyện để thiết lập mặc định thông minh
                    default_val = float(df_main[feat].median())
                    min_val = float(df_main[feat].min())
                    max_val = float(df_main[feat].max())
                    
                    with col_form:
                        inputs[feat] = st.number_input(
                            f"Thông số {feat}",
                            min_value=min_val - 10.0,
                            max_value=max_val + 10.0,
                            value=default_val,
                            format="%.6f"
                        )
                
                submit_pred = st.form_submit_button("🔍 Phân tích Giao dịch", type="primary")
                
                if submit_pred:
                    # Chuyển đổi dict thành dataframe
                    df_inf = pd.DataFrame([inputs])
                    # Chuẩn hóa chuẩn theo scaler đã fit
                    df_inf_scaled = scaler.transform(df_inf)
                    
                    # Tiến hành dự đoán
                    pred_class = model.predict(df_inf_scaled)[0]
                    
                    st.divider()
                    if pred_class == 1:
                        st.error("🚨 **CẢNH BÁO:** Hệ thống nhận diện đây là một **GIAO DỊCH GIAN LẬN** (Rủi ro cao)!")
                    else:
                        st.success("✅ **AN TOÀN:** Hệ thống nhận diện đây là một **GIAO DỊCH HỢP LỆ**.")
                        
                    if hasattr(model, "predict_proba"):
                        pred_prob = model.predict_proba(df_inf_scaled)[0]
                        st.metric("Xác suất gian lận", f"{pred_prob[1]*100:.2f}%")
                        
        elif mode == "Tải lên file dữ liệu tổng hợp (X_new)":
            st.write("##### Tải lên danh sách nhiều giao dịch cần kiểm tra hàng loạt:")
            excel_new_file = st.file_uploader(
                "Chọn tệp Excel hoặc CSV chứa các cột biến đầu vào (X_1 đến X_14):",
                type=["xlsx", "xls", "csv"],
                key="inference_bulk_uploader"
            )
            
            if excel_new_file is not None:
                # Đọc file dữ liệu mới
                if excel_new_file.name.endswith('.csv'):
                    df_new = pd.read_csv(excel_new_file)
                else:
                    df_new = pd.read_excel(excel_new_file)
                
                # Kiểm tra schema biến đầu vào
                missing_inf_cols = [c for c in FEATURES if c not in df_new.columns]
                if missing_inf_cols:
                    st.error(f"Tệp tải lên không hợp lệ. Thiếu các cột đặc trưng sau: {missing_inf_cols}")
                else:
                    # Lấy đúng thứ tự và số lượng cột đặc trưng
                    X_new_data = df_new[FEATURES]
                    X_new_scaled = scaler.transform(X_new_data)
                    
                    # Dự báo hàng loạt
                    bulk_preds = model.predict(X_new_scaled)
                    
                    # Thêm cột kết quả vào DataFrame hiển thị
                    df_result = df_new.copy()
                    df_result['Dự_Báo_Nhãn'] = bulk_preds
                    df_result['Ý_Nghĩa_Kết_Quả'] = df_result['Dự_Báo_Nhãn'].map({0: "Hợp lệ", 1: "Gian lận"})
                    
                    if hasattr(model, "predict_proba"):
                        bulk_probs = model.predict_proba(X_new_scaled)[:, 1]
                        df_result['Xác_Suất_Gian_Lận'] = bulk_probs
                    
                    st.success(f"📈 Đã phân tích thành công {df_result.shape[0]} giao dịch!")
                    
                    # Hiển thị số lượng phát hiện thống kê nhanh
                    fraud_count = int((bulk_preds == 1).sum())
                    st.metric("Số giao dịch nghi ngờ gian lận phát hiện", f"{fraud_count} / {df_result.shape[0]}")
                    
                    # Hiển thị bảng dữ liệu kết quả kèm thanh cuộn gọn gàng
                    st.write("##### Bảng kết quả chi tiết:")
                    st.dataframe(df_result, use_container_width=True)
                    
                    # Tạo nút tải xuống file kết quả dạng CSV
                    csv_data = df_result.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 Tải xuống kết quả phân tích (.CSV)",
                        data=csv_data,
                        file_name="ket_qua_phat_hien_gian_lan.csv",
                        mime="text/csv"
                    )
