# app_klasterisasi.py

import streamlit as st
import pandas as pd
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Segmentasi Karyawan",
    page_icon="📊",
    layout="wide"
)

# --- Fungsi untuk Memuat Data ---
@st.cache_data
def load_data():
    """Memuat laporan, model, dan data mentah untuk scaling."""
    # Memuat laporan
    with open('clustering_models_report.json', 'r') as f:
        report = json.load(f)

    # Mencari path model
    model_dir = 'clustering_models'
    model_files = {f.replace('_model.joblib', ''): os.path.join(model_dir, f) 
                   for f in os.listdir(model_dir) if f.endswith('.joblib')}
                   
    # Memuat data mentah untuk fitting scaler
    try:
        raw_data = pd.read_csv('HR_Analytics.csv')
    except FileNotFoundError:
        st.error("File HR_Analytics.csv tidak ditemukan. File ini diperlukan untuk proses scaling.")
        raw_data = None
    
    return report, model_files, raw_data

# --- Muat Data ---
report, model_files, raw_data = load_data()
if raw_data is None:
    st.stop()

model_names = list(model_files.keys())

# --- Tampilan Utama ---
st.title("📊 Aplikasi Segmentasi Karyawan")
st.write("""
Aplikasi ini menggunakan model Clustering untuk mengelompokkan seorang karyawan ke dalam segmen tertentu. 
Setiap segmen memiliki karakteristik unik yang dapat dilihat pada profil cluster.
""")

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("Masukkan Data Karyawan")

# Pilihan Model
selected_model_name = st.sidebar.selectbox("Pilih Model Clustering", model_names, index=model_names.index("KMeans"))

# Input Fitur
job_satisfaction = st.sidebar.select_slider("Tingkat Kepuasan Kerja", options=[1, 2, 3, 4], value=3)
performance_rating = st.sidebar.select_slider("Peringkat Kinerja", options=[1, 2, 3, 4], value=3)
years_at_company = st.sidebar.slider("Lama Bekerja di Perusahaan (Tahun)", 0, 40, 5)
work_life_balance = st.sidebar.select_slider("Keseimbangan Kerja-Hidup", options=[1, 2, 3, 4], value=3)
monthly_income = st.sidebar.number_input("Pendapatan Bulanan ($)", min_value=1000, max_value=20000, value=5000, step=100)

# Tombol Segmentasi
if st.sidebar.button("📊 Segmentasi Sekarang"):
    
    # 1. Memuat model yang dipilih
    model_path = model_files[selected_model_name]
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan di path: {model_path}")
        st.stop()

    # 2. Menyiapkan data input dan scaling
    features_clu = ['JobSatisfaction', 'PerformanceRating', 'YearsAtCompany', 'WorkLifeBalance', 'MonthlyIncome']
    
    # Buat scaler dan fit dengan data asli
    scaler = StandardScaler()
    scaler.fit(raw_data[features_clu].dropna())
    
    # Siapkan input pengguna
    input_data = pd.DataFrame({
        'JobSatisfaction': [job_satisfaction],
        'PerformanceRating': [performance_rating],
        'YearsAtCompany': [years_at_company],
        'WorkLifeBalance': [work_life_balance],
        'MonthlyIncome': [monthly_income]
    })
    
    # Scale input pengguna menggunakan scaler yang sudah di-fit
    input_scaled = scaler.transform(input_data)
    
    # 3. Melakukan Prediksi Cluster
    cluster_prediction = model.predict(input_scaled)[0]
    
    # 4. Menampilkan Hasil
    st.header(f"Hasil Segmentasi Menggunakan {selected_model_name}")
    st.success(f"Karyawan ini termasuk dalam **Cluster {cluster_prediction}**.", icon="🎯")
    
    st.write("---")
    st.subheader(f"🔍 Profil untuk Cluster {cluster_prediction}")
    
    # Menampilkan profil cluster dari laporan JSON
    model_report = report.get(selected_model_name, {})
    cluster_profiles = model_report.get('cluster_profiles', {})
    
    # Mengonversi key string dari JSON ke integer
    cluster_profiles = {int(k): v for k, v in cluster_profiles.items()}
    
    profile_data = cluster_profiles.get(cluster_prediction)

    if profile_data:
        st.write("Karyawan dalam cluster ini rata-rata memiliki karakteristik sebagai berikut:")
        profile_df = pd.DataFrame.from_dict(profile_data, orient='index', columns=['Rata-rata Nilai'])
        st.dataframe(profile_df)
    else:
        st.warning("Profil untuk cluster ini tidak ditemukan dalam laporan.")


# --- Tampilan Laporan Kinerja ---
st.write("---")
st.header("Laporan & Profil Cluster")
st.write(f"Berikut adalah laporan untuk model **{selected_model_name}**.")

model_report = report.get(selected_model_name, {})

# Menampilkan metrik
col1, col2, col3 = st.columns(3)
col1.metric("Silhouette Score", f"{model_report.get('silhouette_score', 0):.3f}")
col2.metric("Jumlah Cluster Ditemukan", model_report.get('n_clusters_found', 'N/A'))
col3.metric("Jumlah Noise Points", model_report.get('n_noise_points', 'N/A'))

# Menampilkan profil semua cluster
st.subheader("Profil Rata-rata Semua Cluster")
cluster_profiles = model_report.get('cluster_profiles', {})
if cluster_profiles:
    # Mengonversi key string dari JSON ke integer untuk sorting
    cluster_profiles_sorted = {int(k): v for k, v in cluster_profiles.items()}
    profile_all_df = pd.DataFrame.from_dict(cluster_profiles_sorted, orient='index').sort_index()
    profile_all_df.index.name = "Cluster"
    st.dataframe(profile_all_df)
else:
    st.warning("Tidak ada data profil cluster yang ditemukan.")