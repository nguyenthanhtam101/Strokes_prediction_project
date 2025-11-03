# 🧠 Dự án Hệ thống Dự đoán Đột quỵ (Strokes Prediction Project)

[![Streamlit App](httpsS::/static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://strokespredictionproject-lycxr4b2tkapp5ytvl2aksc.streamlit.app/#thong-tin-ho-so-suc-khoe)
* **Notebook Huấn luyện:** [Xem chi tiết trên Google Colab](https://colab.research.google.com/drive/1nmkiLYdhAZvgg7GYAz-h0NQbNBTekZnH?usp=sharing)

---

## 🚀 Giới thiệu chung

Đây là một dự án ứng dụng web được xây dựng bằng Streamlit, nhằm mục đích sàng lọc và hỗ trợ dự đoán nguy cơ đột quỵ dựa trên công nghệ Trí tuệ Nhân tạo (AI).

Điểm đặc biệt của dự án là việc triển khai một hệ thống **hybrid-AI**, kết hợp 3 mô hình machine learning và deep learning khác nhau để phân tích hai nguồn dữ liệu:

1.  **Dữ liệu Y tế (Tabular):** Phân tích các yếu tố rủi ro (tuổi, BMI, đường huyết...) và các triệu chứng lâm sàng.
2.  **Dữ liệu Hình ảnh (Image):** Phân tích hình ảnh CT não để phát hiện dấu hiệu xuất huyết (chảy máu).

Ứng dụng được thiết kế với ba giao diện chính:
* **Tab Bệnh nhân:** Giao diện thân thiện để người dùng tự đánh giá nguy cơ.
* **Tab Bác sĩ:** Một dashboard chuyên sâu, hỗ trợ sàng lọc hàng loạt và giải thích dự đoán bằng SHAP.
* **Tab Chẩn đoán Ảnh:** Giao diện tải lên và phân tích ảnh CT não.

---

## ✨ Các chức năng chính

* **Dự đoán 3 Mô hình:**
    * **Model A (XGBoost):** Phân tích các yếu tố rủi ro sức khỏe nền (tuổi, giới tính, BMI, cao huyết áp, bệnh tim...).
    * **Model B (XGBoost):** Phân tích các triệu chứng cấp tính (rung nhĩ, đau ngực, chóng mặt...).
    * **Model C (Keras/ResNet):** Phân tích ảnh CT não để phân loại (Xuất huyết/Bình thường).
* **Sàng lọc Bệnh nhân Hàng loạt:** (Tab Bác sĩ) Cho phép tải lên file Excel/CSV chứa danh sách bệnh nhân, hệ thống sẽ tự động đánh giá và sắp xếp kết quả theo mức độ rủi ro.
* **Giải thích AI (Explainable AI):** (Tab Bác sĩ) Sử dụng thư viện **SHAP** để tạo biểu đồ waterfall plot, giải thích *lý do tại sao* Model A đưa ra dự đoán nguy cơ cho một bệnh nhân cụ thể.

---

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python (Phiên bản được ghim trên Streamlit là 3.11).
* **Web Framework:** `Streamlit`
* **Mô hình (Training & Inference):**
    * `Scikit-learn` (Xử lý dữ liệu)
    * `XGBoost` (Cho Model A & B)
    * `TensorFlow (2.15.0)` & `Keras` (Cho Model C)
    * `TensorFlow Hub` (Để tải các lớp ResNet)
* **Xử lý dữ liệu/ảnh:** `Pandas`, `Numpy`, `Pillow (PIL)`
* **Giải thích Model:** `SHAP`
* **Lưu trữ Model (Nặng):** **Hugging Face Hub** (Git LFS)
    * Repo code: `github.com/nguyenthanhtam101/Strokes_prediction_project`
    * Repo model: `https://huggingface.co/tam43621/stroke-predict` 

---

## 📂 Cấu trúc Dự án (Các file chức năng)

Dự án này được chia thành hai kho lưu trữ (repository) riêng biệt để vượt qua giới hạn file 100MB của GitHub và tối ưu hóa việc triển khai.

### 1. Repo GitHub (Repo chính - "Nhẹ")

Đây là kho lưu trữ chứa toàn bộ mã nguồn của ứng dụng Streamlit.

* `app.py`:
    * **Chức năng:** File Python chính chứa toàn bộ logic và giao diện người dùng (UI/UX) của ứng dụng Streamlit.
    * Nó chịu trách nhiệm tải các model từ Hugging Face Hub, nhận dữ liệu đầu vào từ người dùng, xử lý, dự đoán và hiển thị kết quả (bao gồm cả các biểu đồ SHAP).

* `requirements.txt`:
    * **Chức năng:** Danh sách các thư viện Python cần thiết (ví dụ: `streamlit`, `pandas`, `tensorflow==2.15.0`, `tensorflow-hub`, `huggingface-hub`).
    * Streamlit Cloud sẽ tự động đọc file này để xây dựng môi trường chạy chính xác.

* `README.md`:
    * **Chức năng:** (Là file này) Cung cấp thông tin tổng quan và hướng dẫn về dự án.

### 2. Repo Hugging Face (Repo Model - "Nặng")

Do các file model (`.keras`, `.json`) quá lớn, chúng được lưu trữ trên Hugging Face Hub (sử dụng Git LFS) và được `app.py` tải về khi khởi động.

* `models/model_A_final.json`:
    * **Chức năng:** Model XGBoost đã huấn luyện cho các yếu tố rủi ro sức khỏe (Dùng cho Tab Bệnh nhân & Bác sĩ).

* `models/model_B_final.json`:
    * **Chức năng:** Model XGBoost đã huấn luyện cho các triệu chứng cấp tính (Dùng cho Tab Bệnh nhân).

* `models/model_C_final.keras` (hoặc `.h5`):
    * **Chức năng:** Model Keras/ResNet đã huấn luyện để phân tích ảnh CT não (Dùng cho Tab Chẩn đoán Ảnh).

* `models/scaler_A_final.pkl` & `models/scaler_B_final.pkl`:
    * **Chức năng:** Các bộ `StandardScaler` đã được "fit" trên dữ liệu huấn luyện. Chúng bắt buộc phải có để chuẩn hóa dữ liệu đầu vào mới của người dùng (từ Tab Bệnh nhân/Bác sĩ) trước khi đưa vào Model A và B.

* `models/columns_A_final.pkl` & `models/columns_B_final.pkl`:
    * **Chức năng:** Lưu danh sách và thứ tự chính xác của các cột (features) mà Model A và B đã học. Điều này đảm bảo dữ liệu đầu vào luôn đúng thứ tự.

* `models/X_train_sample_scaled.pkl`:
    * **Chức năng:** Một mẫu dữ liệu huấn luyện (khoảng 100 dòng) đã được chuẩn hóa. Thư viện SHAP cần file này làm dữ liệu "nền" (background data) để so sánh và giải thích các dự đoán mới.

* `.gitattributes`:
    * **Chức năng:** File cấu hình để báo cho Git LFS biết cần theo dõi và xử lý các file đuôi `.json`, `.pkl`, và `.keras` (vì chúng là file lớn).
