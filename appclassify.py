# app_klasifikasi.py

import streamlit as st
import pandas as pd
import joblib
import json
import os

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Attrition Karyawan",
    page_icon="👤",
    layout="wide"
)

# --- Fungsi untuk Memuat Model dan Laporan ---
@st.cache_data
def load_data():
    """Memuat semua model dan laporan kinerja."""
    # Memuat laporan
    with open('classification_models_report.json', 'r') as f:
        report = json.load(f)

    # Mencari path model
    model_dir = 'classification_models'
    model_files = {f.replace('_model.joblib', ''): os.path.join(model_dir, f) 
                   for f in os.listdir(model_dir) if f.endswith('.joblib')}
    
    return report, model_files

# --- Muat Data ---
report, model_files = load_data()
model_names = list(model_files.keys())

# --- Tampilan Utama ---
st.title("👤 Aplikasi Prediksi Attrition Karyawan")
st.write("""
Aplikasi ini menggunakan model Machine Learning untuk memprediksi kemungkinan seorang karyawan akan mengundurkan diri (attrition). 
Silakan masukkan data karyawan di sidebar kiri dan pilih model yang ingin digunakan.
""")

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("Masukkan Data Karyawan")

# Pilihan Model
selected_model_name = st.sidebar.selectbox("Pilih Model Klasifikasi", model_names, index=model_names.index("RandomForestClassifier"))

# Input Fitur
age = st.sidebar.slider("Usia", 18, 65, 35)
job_satisfaction = st.sidebar.select_slider("Tingkat Kepuasan Kerja", options=[1, 2, 3, 4], value=3)
monthly_income = st.sidebar.number_input("Pendapatan Bulanan ($)", min_value=1000, max_value=20000, value=5000, step=100)
overtime_option = st.sidebar.selectbox("Bekerja Lembur?", ("Ya", "Tidak"))
years_at_company = st.sidebar.slider("Lama Bekerja di Perusahaan (Tahun)", 0, 40, 5)
job_level = st.sidebar.select_slider("Level Jabatan", options=[1, 2, 3, 4, 5], value=2)
env_satisfaction = st.sidebar.select_slider("Tingkat Kepuasan Lingkungan", options=[1, 2, 3, 4], value=3)
distance_from_home = st.sidebar.slider("Jarak dari Rumah (km)", 1, 30, 10)

# Tombol Prediksi
if st.sidebar.button("🔮 Prediksi Sekarang"):
    
    # 1. Memuat model yang dipilih
    model_path = model_files[selected_model_name]
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan di path: {model_path}")
        st.stop()

    # 2. Menyiapkan data input untuk prediksi
    overtime_value = 1 if overtime_option == "Ya" else 0
    
    input_data = pd.DataFrame({
        'Age': [age],
        'JobSatisfaction': [job_satisfaction],
        'MonthlyIncome': [monthly_income],
        'OverTime': [overtime_value],
        'YearsAtCompany': [years_at_company],
        'JobLevel': [job_level],
        'EnvironmentSatisfaction': [env_satisfaction],
        'DistanceFromHome': [distance_from_home]
    })
    
    # 3. Melakukan Prediksi
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # 4. Menampilkan Hasil
    st.header(f"Hasil Prediksi Menggunakan {selected_model_name}")
    
    if prediction == 1: # Asumsi 1 = Yes (Attrition)
        st.warning("**Prediksi: Karyawan Berisiko Mengundurkan Diri (Attrition)**", icon="⚠️")
        st.write(f"Probabilitas Attrition: **{prediction_proba[1]*100:.2f}%**")
    else:
        st.success("**Prediksi: Karyawan Cenderung Bertahan**", icon="✅")
        st.write(f"Probabilitas Bertahan: **{prediction_proba[0]*100:.2f}%**")

    # Menampilkan detail probabilitas
    st.write("---")
    st.write("Detail Probabilitas:")
    st.dataframe(pd.DataFrame({
        'Kategori': ['Bertahan (No)', 'Attrition (Yes)'],
        'Probabilitas': [f"{p*100:.2f}%" for p in prediction_proba]
    }))


# --- Tampilan Laporan Kinerja ---
st.write("---")
st.header("Laporan Kinerja Model")
st.write(f"Berikut adalah laporan kinerja untuk model **{selected_model_name}** yang dievaluasi pada data uji.")

# Menampilkan metrik utama
model_report = report.get(selected_model_name, {})
accuracy = model_report.get('accuracy', 'N/A')
if isinstance(accuracy, float):
    st.metric("Akurasi Model", f"{accuracy*100:.2f}%")

# Menampilkan laporan klasifikasi dalam bentuk tabel
if 'Yes' in model_report and 'No' in model_report:
    report_df = pd.DataFrame({
        'Class': ['No (Bertahan)', 'Yes (Attrition)'],
        'Precision': [model_report['No']['precision'], model_report['Yes']['precision']],
        'Recall': [model_report['No']['recall'], model_report['Yes']['recall']],
        'F1-Score': [model_report['No']['f1-score'], model_report['Yes']['f1-score']],
        'Support': [model_report['No']['support'], model_report['Yes']['support']],
    }).set_index('Class')
    st.dataframe(report_df)
else:
    st.json(model_report)