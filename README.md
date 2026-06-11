# Hệ thống Phát hiện Giao dịch Gian lận trên Nền tảng Streamlit

Ứng dụng web được xây dựng nhằm mục đích chuyển đổi quy trình huấn luyện học máy từ notebook (`phat_hien_giao_dich_gian_lan.ipynb`) thành một công cụ tương tác trực quan. Hệ thống hỗ trợ cán bộ rủi ro tải tập dữ liệu giao dịch mẫu lên, tinh chỉnh siêu tham số, huấn luyện các thuật toán phân loại và dự đoán trực tiếp rủi ro của giao dịch mới.

## 🛠️ Tính năng chính của từng Tab thành phần

1. **📊 Tổng quan dữ liệu**: Kiểm tra kích thước tệp, hiển thị cấu trúc dữ liệu thô đầu vào và tính toán thống kê mô tả tự động (mean, std, min, max, median) cho toàn bộ 14 biến đặc trưng hệ thống.
2. **📈 Trực quan hóa dữ liệu**: Xem biểu đồ phân phối mất cân bằng của nhãn mục tiêu (`default`) và lựa chọn linh hoạt từng thuộc tính số để phân tích phân phối hộp (Box Plot) giữa hai nhóm giao dịch gian lận và hợp lệ.
3. **🔬 Kết quả huấn luyện**: Đánh giá chi tiết hiệu suất của 3 thuật toán tùy chọn cấu hình từ Sidebar (Logistic Regression, Decision Tree, Random Forest) qua các thang đo chuẩn: Accuracy, Precision, Recall, F1-Score kèm Ma trận nhầm lẫn trực quan.
4. **🔮 Sử dụng mô hình**: Cung cấp hai chế độ dự đoán mạnh mẽ:
   - Nhập thủ công chỉ số của 1 giao dịch đơn lẻ qua biểu mẫu thông minh tự gán giá trị mặc định là trung vị dữ liệu gốc.
   - Tải lên file Excel/CSV danh sách hàng loạt giao dịch mới (`X_new`) để quét rủi ro đồng loạt và xuất báo cáo kết quả định dạng `.csv`.

## 📦 Hướng dẫn cài đặt và khởi chạy

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo máy tính của bạn đã được cài đặt sẵn Python (Khuyến nghị phiên bản từ `3.10` đến `3.12`).

### Bước 2: Cài đặt các thư viện phụ thuộc bắt buộc
Mở terminal tại thư mục chứa mã nguồn và thực thi lệnh sau:
```bash
pip install -r requirements.txt
