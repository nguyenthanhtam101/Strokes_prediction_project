import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import io 
from PIL import Image 
import tensorflow as tf 
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download 
import tensorflow_hub as hub 
from tensorflow.keras.utils import custom_object_scope # <-- THÊM DÒNG NÀY

# --- 1. CẤU HÌNH TRANG VÀ TẢI MÔ HÌNH ---

st.set_page_config( page_title="Hệ Thống Dự Đoán Đột Quỵ", page_icon="🧠", layout="wide")

# (Giữ nguyên các dòng HF_REPO_ID và FILENAME...)
HF_REPO_ID = "tam43621/stroke-predict" 
MODEL_PATH = "models/" 
MODEL_A_FILENAME = MODEL_PATH + "model_A_final.json"
SCALER_A_FILENAME = MODEL_PATH + "scaler_A_final.pkl"
COLS_A_FILENAME = MODEL_PATH + "columns_A_final.pkl"
MODEL_B_FILENAME = MODEL_PATH + "model_B_final.json"
SCALER_B_FILENAME = MODEL_PATH + "scaler_B_final.pkl"
COLS_B_FILENAME = MODEL_PATH + "columns_B_final.pkl"
X_TRAIN_SAMPLE_FILENAME = MODEL_PATH + "X_train_sample_scaled.pkl"
MODEL_C_FILENAME = MODEL_PATH + "model2_C_resnet.h5" 

@st.cache_resource
def load_models_and_data():
    """Tải 3 model, scaler, cột từ Hugging Face Hub."""
    try:
        # Tải từng file từ Hugging Face
        model_a_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_A_FILENAME)
        model_b_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_B_FILENAME)
        model_c_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_C_FILENAME)
        scaler_a_path = hf_hub_download(repo_id=HF_REPO_ID, filename=SCALER_A_FILENAME)
        scaler_b_path = hf_hub_download(repo_id=HF_REPO_ID, filename=SCALER_B_FILENAME)
        cols_a_path = hf_hub_download(repo_id=HF_REPO_ID, filename=COLS_A_FILENAME)
        cols_b_path = hf_hub_download(repo_id=HF_REPO_ID, filename=COLS_B_FILENAME)
        train_sample_path = hf_hub_download(repo_id=HF_REPO_ID, filename=X_TRAIN_SAMPLE_FILENAME)

        # Tải model A và B
        model_a = xgb.XGBClassifier(); model_a.load_model(model_a_path)
        model_b = xgb.XGBClassifier(); model_b.load_model(model_b_path)
        
        # --- SỬA LỖI MODEL C (Dùng custom_object_scope) ---
        # Báo cho Keras biết về các lớp của TensorFlow Hub
        # Chúng ta dùng 'with' (context manager) thay vì truyền dict
        with custom_object_scope({'KerasLayer': hub.KerasLayer}):
             model_c = load_model(model_c_path, compile=False)
        # --- KẾT THÚC SỬA LỖI ---
        
        train_sample_scaled = joblib.load(train_sample_path)
        cols_a = joblib.load(cols_a_path); cols_b = joblib.load(cols_b_path)
        
        if not isinstance(train_sample_scaled, pd.DataFrame): train_sample_scaled = pd.DataFrame(train_sample_scaled, columns=cols_a)
        elif list(train_sample_scaled.columns) != list(cols_a): train_sample_scaled.columns = cols_a

        models_data = {
            "model_A": model_a, "scaler_A": joblib.load(scaler_a_path), "cols_A": cols_a,
            "model_B": model_b, "scaler_B": joblib.load(scaler_b_path), "cols_B": cols_b,
            "model_C": model_c,
            "train_sample_scaled": train_sample_scaled
        }
        print("Đã tải 3 model và dữ liệu mẫu từ Hugging Face thành công.")
        return models_data
    except Exception as e: st.error(f"Lỗi khi tải model từ Hugging Face: {e}"); st.exception(e); return None

models_data = load_models_and_data()
if models_data is None: st.warning("Không tải được model."); st.stop()
numerical_cols_s = ['age', 'avg_glucose_level', 'bmi']
# --- Giả định kích thước ảnh Model C (thay đổi nếu cần) ---
IMG_SIZE = (224, 224) 


# --- HÀM LOGIC CHO MODEL A & B (Giữ nguyên) ---
def predict_final_risk_v3(patient_health_df, patient_symptoms_df):
    health_original = patient_health_df.copy(); symptoms_original = patient_symptoms_df.copy()
    age=health_original.get('age', pd.Series([0])).iloc[0]; bmi=health_original.get('bmi', pd.Series([np.nan])).iloc[0]
    glucose=health_original.get('avg_glucose_level', pd.Series([0])).iloc[0]; hypertension=health_original.get('hypertension', pd.Series([0])).iloc[0]
    heart_disease=health_original.get('heart_disease', pd.Series([0])).iloc[0]; irregular_heartbeat=symptoms_original.get('Irregular Heartbeat', pd.Series([0])).iloc[0]
    bmi_value_for_check = bmi if pd.notna(bmi) else 0

    if pd.notna(age) and pd.notna(irregular_heartbeat) and age > 65 and irregular_heartbeat == 1: return "CAO (Dựa trên rung nhĩ và tuổi tác)", 0.95, "red_flag"
    if pd.notna(age) and pd.notna(glucose) and pd.notna(hypertension) and pd.notna(heart_disease) and \
       age < 40 and bmi_value_for_check < 30 and glucose < 140 and hypertension == 0 and heart_disease == 0: return "THẤP (Dựa trên kiến thức y khoa)", 0.05, "green_flag"

    model_A = models_data["model_A"]; model_B = models_data["model_B"]; scaler_A = models_data["scaler_A"]
    scaler_B = models_data["scaler_B"]; columns_A = models_data["cols_A"]; columns_B = models_data["cols_B"]
    health_df_processed = health_original.reindex(columns=columns_A, fill_value=0)
    for col in numerical_cols_s: health_df_processed[col] = pd.to_numeric(health_df_processed[col], errors='coerce')
    mean_values = models_data["train_sample_scaled"][numerical_cols_s].mean() if models_data["train_sample_scaled"] is not None else 0
    health_df_processed = health_df_processed.fillna(mean_values)
    prob_A = 0.0
    try:
        health_df_processed[numerical_cols_s] = health_df_processed[numerical_cols_s].astype(float)
        health_df_processed[numerical_cols_s] = scaler_A.transform(health_df_processed[numerical_cols_s])
        prob_A = model_A.predict_proba(health_df_processed)[:, 1][0]
    except Exception as e_scale_A: st.warning(f"Lỗi scale A: {e_scale_A}")

    symptoms_df_scaled = symptoms_original.copy()
    if 'age' in symptoms_df_scaled.columns and 'Age' not in symptoms_df_scaled.columns: symptoms_df_scaled=symptoms_df_scaled.rename(columns={'age': 'Age'})
    prob_B = 0.0
    if 'Age' in symptoms_df_scaled.columns and pd.to_numeric(symptoms_df_scaled['Age'], errors='coerce').notna().all():
        symptoms_df_scaled['Age'] = pd.to_numeric(symptoms_df_scaled['Age'], errors='coerce')
        try:
            symptoms_df_scaled[['Age']] = symptoms_df_scaled[['Age']].astype(float) # Đảm bảo kiểu float
            symptoms_df_scaled[['Age']] = scaler_B.transform(symptoms_df_scaled[['Age']])
            symptoms_for_predict = symptoms_df_scaled.reindex(columns=columns_B, fill_value=0)
            prob_B = model_B.predict_proba(symptoms_for_predict)[:, 1][0]
        except Exception as e_scale_B: st.warning(f"Lỗi scale B: {e_scale_B}")

    if prob_B >= 0.5: return "CAO (Dựa trên triệu chứng cấp tính)", prob_B, "ai_model_b"
    elif prob_A >= 0.2: return "TRUNG BÌNH (Dựa trên yếu tố rủi ro tiềm ẩn)", prob_A, "ai_model_a"
    else: return "THẤP", max(prob_A, prob_B), "ai_low"

# --- 3. GIAO DIỆN ỨNG DỤNG ---

st.title("🧠 Hệ Thống Sàng Lọc & Dự Đoán Đột Quỵ (3 Model)")
st.markdown("Ứng dụng kết hợp AI (tabular, hình ảnh) và logic y khoa để đánh giá nguy cơ đột quỵ.")

# --- CẬP NHẬT: Thêm Tab 3 cho Model C ---
tab_names = ["Dành cho Bệnh nhân (Model A+B)", "Dành cho Bác sĩ (Model A+B)", "Chẩn đoán Hình ảnh (Model C)"]
tab_patient, tab_doctor, tab_image = st.tabs(tab_names)

# --- TAB DÀNH CHO BỆNH NHÂN (Giữ nguyên) ---
with tab_patient:
    st.header("Công Cụ Tự Đánh Giá Nguy Cơ")
    st.write("Vui lòng cung cấp các thông tin dưới đây để hệ thống phân tích.")
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Thông tin Hồ sơ Sức khỏe")
            age = st.number_input("Tuổi của bạn", min_value=1, max_value=120, value=None, step=1, placeholder="Nhập tuổi...", key="p_age")
            gender_selected = st.radio("Giới tính", ["Nam", "Nữ"], index=None, key="p_gender")
            avg_glucose_level = st.number_input("Mức đường huyết TB (mg/dL)", min_value=50.0, max_value=300.0, value=None, step=0.1, format="%.1f", placeholder="Ví dụ: 100.0", key="p_glucose")
            bmi = st.number_input("Chỉ số BMI", min_value=10.0, max_value=70.0, value=None, step=0.1, format="%.1f", placeholder="Ví dụ: 22.5", key="p_bmi")
            hypertension = 1 if st.checkbox("Cao huyết áp?", key="p_ht") else 0
            heart_disease = 1 if st.checkbox("Bệnh tim?", key="p_hd") else 0
            smoking_status_options = ["Chưa bao giờ hút", "Đã từng hút", "Đang hút thuốc", "Không rõ"]
            smoking_status = st.selectbox("Tình trạng hút thuốc", smoking_status_options, index=None, placeholder="Chọn tình trạng...", key="p_smoke")
            smoking_formerly = 1 if smoking_status == "Đã từng hút" else 0
            smoking_never = 1 if smoking_status == "Chưa bao giờ hút" else 0
            smoking_smokes = 1 if smoking_status == "Đang hút thuốc" else 0
        with col2:
            st.subheader("Thông tin Triệu chứng (nếu có)")
            symptoms = {}
            symptom_columns = models_data["cols_B"].tolist(); symptom_columns.remove('Age')
            symptom_translation = {
                'Irregular Heartbeat': 'Nhịp tim không đều (Rung nhĩ)', 'High Blood Pressure': 'Huyết áp cao (triệu chứng)',
                'Chest Pain': 'Đau ngực', 'Shortness of Breath': 'Khó thở', 'Dizziness': 'Chóng mặt, xây xẩm',
                'Fatigue & Weakness': 'Mệt mỏi & Yếu cơ', 'Swelling (Edema)': 'Phù (sưng) tay chân',
                'Pain in Neck/Jaw/Shoulder/Back': 'Đau cổ/hàm/vai/lưng', 'Excessive Sweating': 'Đổ mồ hôi nhiều',
                'Persistent Cough': 'Ho dai dẳng', 'Nausea/Vomiting': 'Buồn nôn/Nôn',
                'Chest Discomfort (Activity)': 'Khó chịu ở ngực (khi hoạt động)', 'Cold Hands/Feet': 'Lạnh tay/chân',
                'Snoring/Sleep Apnea': 'Ngáy/Ngưng thở khi ngủ', 'Anxiety/Feeling of Doom': 'Lo lắng/Cảm giác bất an'
            }
            for symptom_name in symptom_columns:
                label = symptom_translation.get(symptom_name, symptom_name.replace('_', ' ').title())
                key = f"p_sym_{symptom_name}"; symptoms[symptom_name] = 1 if st.checkbox(label, key=key) else 0
        submitted = st.form_submit_button("BẮT ĐẦU DỰ ĐOÁN")
    if submitted:
        if age is None or gender_selected is None or avg_glucose_level is None or bmi is None or smoking_status is None:
            st.error("Vui lòng nhập đầy đủ thông tin Hồ sơ Sức khỏe.")
        else:
            with st.spinner("Hệ thống đang phân tích..."):
                gender_Male = 1 if gender_selected == "Nam" else 0
                health_data = {'age': [age], 'avg_glucose_level': [avg_glucose_level], 'bmi': [bmi], 'hypertension': [hypertension], 'heart_disease': [heart_disease], 'gender_Male': [gender_Male], 'smoking_status_formerly smoked': [smoking_formerly], 'smoking_status_never smoked': [smoking_never], 'smoking_status_smokes': [smoking_smokes]}
                patient_health_df = pd.DataFrame(health_data)
                patient_symptoms_df = pd.DataFrame([symptoms]); patient_symptoms_df['Age'] = age; patient_symptoms_df['age'] = age
                risk_level, probability, source = predict_final_risk_v3(patient_health_df, patient_symptoms_df)
                st.subheader("Kết Quả Phân Tích")
                if "CAO" in risk_level: st.error(f"**Nguy cơ: {risk_level}** ({probability*100:.2f}%)")
                elif "TRUNG BÌNH" in risk_level: st.warning(f"**Nguy cơ: {risk_level}** ({probability*100:.2f}%)")
                else: st.success(f"**Nguy cơ: {risk_level}** ({probability*100:.2f}%)")
                if "CAO" in risk_level: st.warning("Cảnh báo: Nguy cơ cao. Vui lòng liên hệ cơ sở y tế gần nhất.")
                elif "TRUNG BÌNH" in risk_level: st.info("Khuyến nghị: Có yếu tố rủi ro. Duy trì lối sống lành mạnh, theo dõi sức khỏe.")
                else: st.info("Khuyến nghị: Không phát hiện nguy cơ rõ ràng. Tiếp tục duy trì lối sống lành mạnh.")

# --- TAB DÀNH CHO BÁC SĨ (Giữ nguyên) ---
with tab_doctor:
    st.header("Dashboard Hỗ Trợ Chẩn Đoán")
    st.subheader("1. Sàng lọc Bệnh nhân hàng loạt")
    uploaded_file = st.file_uploader("Tải lên file Excel/CSV danh sách bệnh nhân", type=["csv", "xlsx"], key="d_uploader")
    COLUMN_MAP = {
        'age': ['age', 'tuổi', 'tuoi'], 'avg_glucose_level': ['avg_glucose_level', 'glucose', 'đường huyết', 'duong huyet'],
        'bmi': ['bmi', 'chỉ số bmi', 'chiso bmi'], 'hypertension': ['hypertension', 'cao huyết áp', 'huyết áp cao', 'ha', 'tang huyet ap'],
        'heart_disease': ['heart_disease', 'bệnh tim', 'benh tim'], 'gender': ['gender', 'giới tính', 'phái', 'gioi tinh'],
        'smoking_status': ['smoking_status', 'hút thuốc', 'hut thuoc', 'tinh trang hut thuoc'],
        'Irregular Heartbeat': ['irregular heartbeat', 'nhịp tim không đều', 'rung nhĩ', 'nhip tim khong deu'],
        'High Blood Pressure': ['high blood pressure', 'huyết áp cao (triệu chứng)', 'huyet ap cao'], 'Chest Pain': ['chest pain', 'đau ngực', 'dau nguc'],
        'Shortness of Breath': ['shortness of breath', 'khó thở', 'kho tho'], 'Dizziness': ['dizziness', 'chóng mặt', 'chong mat'],
        'Fatigue & Weakness': ['fatigue & weakness', 'mệt mỏi', 'met moi', 'yếu cơ', 'yeu co'], 'Swelling (Edema)': ['swelling (edema)', 'phù', 'sưng', 'phu'],
        'Pain in Neck/Jaw/Shoulder/Back': ['pain in neck/jaw/shoulder/back', 'đau cổ vai gáy', 'dau co vai gay'], 'Excessive Sweating': ['excessive sweating', 'đổ mồ hôi', 'do mo hoi'],
        'Persistent Cough': ['persistent cough', 'ho dai dẳng', 'ho keo dai'], 'Nausea/Vomiting': ['nausea/vomiting', 'buồn nôn', 'non'],
        'Chest Discomfort (Activity)': ['chest discomfort (activity)', 'khó chịu ngực', 'kho chiu nguc'], 'Cold Hands/Feet': ['cold hands/feet', 'lạnh tay chân', 'lanh tay chan'],
        'Snoring/Sleep Apnea': ['snoring/sleep apnea', 'ngáy', 'ngưng thở khi ngủ'], 'Anxiety/Feeling of Doom': ['anxiety/feeling of doom', 'lo lắng', 'bất an'],
        'id': ['id', 'mã bệnh nhân', 'mã bn', 'patient id'], 'name': ['name', 'họ tên', 'tên', 'ten benh nhan']
    }

    def find_col_name(df_columns, synonym_list):
        for name in synonym_list:
            if name in df_columns: return name
        return None

    if 'processed_df' not in st.session_state: st.session_state['processed_df'] = None
    if 'shap_data_dict' not in st.session_state: st.session_state['shap_data_dict'] = {}
    if 'original_df_for_display' not in st.session_state: st.session_state['original_df_for_display'] = None

    if uploaded_file:
        with st.spinner("Đang xử lý file..."):
            try:
                bytes_data = uploaded_file.getvalue(); encodings_to_try = ['utf-8', 'cp1258', 'latin1']; df_patients = None
                for enc in encodings_to_try:
                    try:
                        string_io = io.StringIO(bytes_data.decode(enc))
                        df_patients = pd.read_csv(string_io, na_values=['N/A', 'NA', '']) if uploaded_file.name.endswith('csv') else pd.read_excel(io.BytesIO(bytes_data), na_values=['N/A', 'NA', ''])
                        break
                    except UnicodeDecodeError: continue
                    except Exception as read_err: st.error(f"Lỗi đọc file: {read_err}"); st.stop()
                if df_patients is None: st.error("Không thể đọc file encoding."); st.stop()

                df_original_copy = df_patients.copy() 
                df_patients_normalized = df_patients.copy(); df_patients_normalized.columns = df_patients_normalized.columns.str.lower().str.strip()
                uploaded_cols_normalized = df_patients_normalized.columns.tolist()
                health_cols_needed = models_data["cols_A"].tolist(); symptoms_cols_needed = models_data["cols_B"].tolist()
                results = []; processed_rows_for_shap = {}
                age_col_name = find_col_name(uploaded_cols_normalized, COLUMN_MAP['age'])
                if age_col_name is None: st.error("Lỗi: File thiếu cột 'age'/'tuổi'."); st.stop()
                id_col_name = find_col_name(uploaded_cols_normalized, COLUMN_MAP.get('id', ['id'])) 

                for index, row in df_patients_normalized.iterrows(): 
                    patient_key = index 
                    row_age = pd.to_numeric(row[age_col_name], errors='coerce')
                    if pd.isna(row_age): results.append({"Nguy cơ": f"Lỗi: Tuổi", "Xác suất": 0}); continue

                    health_data = {'age': row_age}; glucose_col = find_col_name(uploaded_cols_normalized, COLUMN_MAP['avg_glucose_level'])
                    health_data['avg_glucose_level'] = pd.to_numeric(row.get(glucose_col, np.nan), errors='coerce')
                    bmi_col = find_col_name(uploaded_cols_normalized, COLUMN_MAP['bmi'])
                    health_data['bmi'] = pd.to_numeric(row.get(bmi_col, np.nan), errors='coerce')
                    ht_col = find_col_name(uploaded_cols_normalized, COLUMN_MAP['hypertension'])
                    health_data['hypertension'] = 1 if ht_col and str(row.get(ht_col, 0)).lower() in ['yes', 'có', '1'] else 0
                    hd_col = find_col_name(uploaded_cols_normalized, COLUMN_MAP['heart_disease'])
                    health_data['heart_disease'] = 1 if hd_col and str(row.get(hd_col, 0)).lower() in ['yes', 'có', '1'] else 0
                    gender_col = find_col_name(uploaded_cols_normalized, COLUMN_MAP['gender'])
                    gender_val = row.get(gender_col) if gender_col else None
                    health_data['gender_Male'] = 1 if (gender_val and str(gender_val).lower() in ['nam', 'male', 'm']) else 0
                    smoking_col = find_col_name(uploaded_cols_normalized, COLUMN_MAP['smoking_status'])
                    smoking_val = row.get(smoking_col) if smoking_col else None
                    health_data['smoking_status_formerly smoked']=1 if (smoking_val and str(smoking_val).lower() in ['đã từng hút', 'formerly smoked','da tung hut']) else 0
                    health_data['smoking_status_never smoked']=1 if (smoking_val and str(smoking_val).lower() in ['chưa bao giờ hút', 'never smoked','chua bao gio hut']) else 0
                    health_data['smoking_status_smokes']=1 if (smoking_val and str(smoking_val).lower() in ['đang hút thuốc', 'smokes','dang hut thuoc']) else 0
                    health_df = pd.DataFrame([health_data])

                    symptoms_data = {'Age': row_age}
                    for col_name_model in symptoms_cols_needed:
                        if col_name_model == 'Age': continue
                        synonyms = COLUMN_MAP.get(col_name_model, [col_name_model.lower().strip()])
                        col_name_upload = find_col_name(uploaded_cols_normalized, synonyms)
                        raw_value = row.get(col_name_upload, 0) if col_name_upload else 0
                        symptoms_data[col_name_model] = 1 if str(raw_value).lower() in ['yes', 'có', '1'] else 0
                    symptoms_df = pd.DataFrame([symptoms_data])
                    symptoms_df['age'] = row_age

                    try:
                        mean_values = models_data["train_sample_scaled"][numerical_cols_s].mean() if models_data["train_sample_scaled"] is not None else 0
                        health_df_filled = health_df.fillna(mean_values) 
                        risk, prob, _ = predict_final_risk_v3(health_df_filled.copy(), symptoms_df.copy())
                        results.append({"Nguy cơ": risk, "Xác suất": prob})
                        processed_rows_for_shap[patient_key] = health_df_filled 
                    except Exception as pred_e:
                        results.append({"Nguy cơ": f"Lỗi dự đoán", "Xác suất": 0})
                        st.warning(f"Lỗi dự đoán BN index {index}: {pred_e}")

                df_results = pd.DataFrame(results)
                df_final = pd.concat([df_original_copy.reset_index(drop=True), df_results.reset_index(drop=True)], axis=1)
                df_final = df_final.sort_values(by="Xác suất", ascending=False)
                st.subheader("Kết quả Sàng lọc (Đã sắp xếp ưu tiên)")
                st.dataframe(df_final)

                st.session_state['processed_df'] = df_final
                st.session_state['shap_data_dict'] = processed_rows_for_shap
                st.session_state['original_df_for_display'] = df_original_copy 
            except Exception as e:
                st.error(f"Đã xảy ra lỗi nghiêm trọng khi xử lý file: {e}")
                st.exception(e)
                st.info("Mẹo: Đảm bảo file CSV/Excel hợp lệ và có cột 'age'/'tuổi'.")
                st.session_state['processed_df'] = None
                st.session_state['shap_data_dict'] = {}
                st.session_state['original_df_for_display'] = None

    # --- PHẦN 2: GIẢI THÍCH SHAP TƯƠNG TÁC (Giữ nguyên) ---
    st.subheader("2. Giải thích Ca bệnh (SHAP - Model A)")
    if st.session_state.get('processed_df') is not None:
        df_display = st.session_state['processed_df']
        shap_data_dict = st.session_state['shap_data_dict']
        df_original_display = st.session_state['original_df_for_display']

        id_col_orig = find_col_name(df_original_display.columns, COLUMN_MAP.get('id', ['id']))
        name_col_orig = find_col_name(df_original_display.columns, COLUMN_MAP.get('name', ['name']))
        display_options = []; option_to_index_map = {}
        for index in df_display.index: 
            label = f"Hàng {index+2}" # Nhãn mặc định
            if id_col_orig and pd.notna(df_original_display.loc[index, id_col_orig]):
                label += f" (ID: {df_original_display.loc[index, id_col_orig]})"
            if name_col_orig and pd.notna(df_original_display.loc[index, name_col_orig]):
                 label += f" - {df_original_display.loc[index, name_col_orig]}"
            display_options.append(label)
            option_to_index_map[label] = index 

        selected_display_option = st.selectbox(
            "Chọn bệnh nhân để giải thích:",
            options=[""] + display_options, index=0, key="d_shap_select"
        )
        if selected_display_option:
            selected_index = option_to_index_map[selected_display_option] 
            patient_info_original = df_original_display.iloc[[selected_index]]
            st.write("Thông tin bệnh nhân đã chọn (dữ liệu gốc):")
            st.dataframe(patient_info_original)
            patient_health_data_for_shap = shap_data_dict.get(selected_index) 
            if patient_health_data_for_shap is None:
                st.warning(f"Không tìm thấy dữ liệu đã xử lý cho bệnh nhân tại index {selected_index}.")
            elif st.button(f"Chạy giải thích cho {selected_display_option}", key="d_shap_run_selected"):
                with st.spinner("Đang tính toán SHAP (có thể mất vài giây)..."):
                    try:
                        model_A = models_data["model_A"]; scaler_A = models_data["scaler_A"]
                        columns_A = models_data["cols_A"]; train_sample = models_data["train_sample_scaled"]
                        if train_sample is None: st.error("Thiếu dữ liệu mẫu."); st.stop()

                        patient_processed = patient_health_data_for_shap.reindex(columns=columns_A, fill_value=0)
                        for col in numerical_cols_s: patient_processed[col] = pd.to_numeric(patient_processed[col], errors='coerce')
                        mean_vals_shap = train_sample[numerical_cols_s].mean()
                        patient_processed = patient_processed.fillna(mean_vals_shap)
                        patient_processed[numerical_cols_s] = scaler_A.transform(patient_processed[numerical_cols_s])

                        def predict_proba_A(data):
                            if not isinstance(data, pd.DataFrame): data_df = pd.DataFrame(data, columns=columns_A)
                            else: data_df = data.copy()
                            for col in numerical_cols_s: data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
                            data_df = data_df.fillna(mean_vals_shap)
                            data_reindexed = data_df.reindex(columns=columns_A, fill_value=0)
                            data_reindexed[numerical_cols_s] = data_reindexed[numerical_cols_s].astype(float)
                            data_reindexed[numerical_cols_s] = scaler_A.transform(data_reindexed[numerical_cols_s])
                            return model_A.predict_proba(data_reindexed)[:, 1]

                        explainer_background = shap.sample(train_sample, min(50, len(train_sample)))
                        if not isinstance(explainer_background, pd.DataFrame): explainer_background = pd.DataFrame(explainer_background, columns=columns_A)
                        explainer = shap.KernelExplainer(predict_proba_A, explainer_background)
                        shap_values = explainer.shap_values(patient_processed.to_numpy())
                        st.write("Biểu đồ SHAP giải thích các yếu tố ảnh hưởng:")
                        fig, ax = plt.subplots()
                        feature_names_vietnamese = { 
                            'age': 'Tuổi', 'avg_glucose_level': 'Đường huyết TB', 'bmi': 'Chỉ số BMI',
                            'hypertension': 'Cao huyết áp (Nền)', 'heart_disease': 'Bệnh tim (Nền)',
                            'gender_Male': 'Giới tính Nam', 'smoking_status_formerly smoked': 'Đã từng hút',
                            'smoking_status_never smoked': 'Chưa bao giờ hút', 'smoking_status_smokes': 'Đang hút thuốc'
                        }
                        display_feature_names = [feature_names_vietnamese.get(col, col) for col in columns_A]
                        expected_value_shap = explainer.expected_value
                        if isinstance(expected_value_shap, (np.ndarray, list)): expected_value_shap = expected_value_shap[0]
                        shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=expected_value_shap,
                                                             data=patient_processed.iloc[0].to_numpy(), feature_names=display_feature_names))
                        st.pyplot(fig, bbox_inches='tight'); plt.close(fig)
                        st.info(""" **Cách đọc biểu đồ:** (Giữ nguyên) """)
                    except Exception as shap_e:
                        st.error(f"Lỗi khi tính toán hoặc vẽ SHAP: {shap_e}"); st.exception(shap_e)
    else:
        st.info("Vui lòng tải lên và xử lý file danh sách bệnh nhân để kích hoạt tính năng giải thích.")


# --- PHẦN MỚI: TAB DÀNH CHO MODEL C (HÌNH ẢNH) ---
with tab_image:
    st.header("Model C: Phân tích Hình ảnh Y khoa (CT Não)")
    st.info("Tải lên ảnh CT não để mô hình phân tích (Hemorrhagic - Chảy máu vs Normal - Bình thường).")

    img_file = st.file_uploader("Tải lên ảnh (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], key="c_uploader")

    if img_file:
        st.image(img_file, caption="Ảnh đã tải lên.", use_column_width=True)

        if st.button("Bắt đầu Phân tích Ảnh", key="c_run"):
            with st.spinner("Đang xử lý ảnh... (Có thể mất một lúc)"):
                try:
                    model_C = models_data["model_C"]

                    # 1. Đọc và tiền xử lý ảnh
                    image = Image.open(img_file).convert('RGB')
                    
                    # 2. Resize về kích thước model C được huấn luyện (Giả định 224x224)
                    # Hãy thay đổi IMG_SIZE ở đầu file nếu model của bạn dùng kích thước khác
                    image_resized = image.resize(IMG_SIZE) 
                    
                    # 3. Chuẩn hóa (Giả định chuẩn hóa về [0, 1])
                    img_array = np.array(image_resized)
                    img_array_normalized = img_array / 255.0 
                    
                    # 4. Tạo batch (1, 224, 224, 3)
                    img_batch = np.expand_dims(img_array_normalized, axis=0)

                    # 5. Dự đoán
                    prediction = model_C.predict(img_batch)
                    prob = prediction[0][0] # Lấy xác suất từ neuron output

                    st.subheader("Kết quả Phân tích Hình ảnh:")
                    
                    # QUAN TRỌNG: Dựa trên code cũ của bạn:
                    # Lớp 0: Hemorrhagic
                    # Lớp 1: NORMAL
                    # Model (binary) dự đoán xác suất của Lớp 1 (NORMAL)
                    
                    if prob > 0.5:
                        st.success(f"**Kết luận: NORMAL (Bình thường)**")
                        st.progress(prob)
                        st.write(f"Độ chắc chắn (Normal): {prob*100:.2f}%")
                    else:
                        st.error(f"**Kết luận: HEMORRHAGIC (Chảy máu)**")
                        st.progress(1.0 - prob)
                        st.write(f"Độ chắc chắn (Hemorrhagic): {(1-prob)*100:.2f}%")
                        st.warning("Cảnh báo: Phát hiện dấu hiệu chảy máu. Cần xem xét y tế ngay lập tức.")

                except Exception as e:
                    st.error(f"Lỗi khi phân tích ảnh: {e}")
                    st.exception(e)