# app_klasterisasi.py (dengan visualisasi)

import streamlit as st
import pandas as pd
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Segmentasi Karyawan (dengan Visualisasi)",
    page_icon="✨",
    layout="wide"
)

# --- Fungsi untuk Memuat Data ---
@st.cache_data
def load_data():
    """Memuat laporan, model, dan data mentah."""
    # Memuat laporan
    with open('clustering_models_report.json', 'r') as f:
        report = json.load(f)

    # Mencari path model
    model_dir = 'clustering_models'
    model_files = {f.replace('_model.joblib', ''): os.path.join(model_dir, f)
                   for f in os.listdir(model_dir) if f.endswith('.joblib')}
                   
    # Memuat data mentah
    try:
        raw_data = pd.read_csv('HR_Analytics.csv')
    except FileNotFoundError:
        st.error("File HR_Analytics.csv tidak ditemukan. File ini diperlukan untuk proses scaling dan visualisasi.")
        raw_data = None
    
    return report, model_files, raw_data

# --- Muat Data ---
report, model_files, raw_data = load_data()
if raw_data is None:
    st.stop()

model_names = list(model_files.keys())

# --- Tampilan Utama ---
st.title("✨ Aplikasi Segmentasi Karyawan dengan Visualisasi")
st.write("""
Aplikasi ini menggunakan model Clustering untuk mengelompokkan karyawan ke dalam segmen tertentu. 
Setelah segmentasi, posisi karyawan akan divisualisasikan di antara karyawan lainnya.
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
if st.sidebar.button("📊 Segmentasi & Visualisasikan"):
    
    # --- Blok Utama untuk Pemrosesan dan Visualisasi ---
    st.header(f"Hasil Analisis Menggunakan {selected_model_name}")

    # 1. Memuat model yang dipilih
    model_path = model_files[selected_model_name]
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan di path: {model_path}")
        st.stop()

    # 2. Menyiapkan data
    features_clu = ['JobSatisfaction', 'PerformanceRating', 'YearsAtCompany', 'WorkLifeBalance', 'MonthlyIncome']
    df_cluster_data = raw_data[features_clu].dropna().copy()
    
    # 3. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster_data)
    
    # 4. Melakukan prediksi cluster pada seluruh data untuk visualisasi
    all_labels = model.fit_predict(X_scaled)
    df_cluster_data['Cluster'] = all_labels
    df_cluster_data['Cluster'] = df_cluster_data['Cluster'].astype(str) # Ubah ke string untuk pewarnaan di plotly

    # 5. Reduksi Dimensi dengan PCA untuk visualisasi
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_cluster_data['PC1'] = components[:, 0]
    df_cluster_data['PC2'] = components[:, 1]
    
    # 6. Memproses Input Pengguna
    input_df = pd.DataFrame({
        'JobSatisfaction': [job_satisfaction],
        'PerformanceRating': [performance_rating],
        'YearsAtCompany': [years_at_company],
        'WorkLifeBalance': [work_life_balance],
        'MonthlyIncome': [monthly_income]
    })
    input_scaled = scaler.transform(input_df)
    
    # 7. Memprediksi Cluster & Mentransformasi PCA untuk Input Pengguna
    cluster_prediction = model.predict(input_scaled)[0]
    input_pca = pca.transform(input_scaled)

    # 8. Menampilkan Hasil Prediksi
    st.success(f"Karyawan ini termasuk dalam **Cluster {cluster_prediction}**.", icon="🎯")
    
    # Menampilkan profil cluster
    model_report = report.get(selected_model_name, {})
    cluster_profiles = {int(k): v for k, v in model_report.get('cluster_profiles', {}).items()}
    profile_data = cluster_profiles.get(cluster_prediction)
    if profile_data:
        st.write(f"**Profil untuk Cluster {cluster_prediction}:** Karyawan dalam cluster ini rata-rata memiliki karakteristik sebagai berikut:")
        profile_df = pd.DataFrame.from_dict(profile_data, orient='index', columns=['Rata-rata Nilai'])
        st.dataframe(profile_df)
    
    # 9. Membuat dan Menampilkan Visualisasi Plotly
    st.write("---")
    st.subheader("Visualisasi Posisi Karyawan dalam Cluster")
    st.write("Grafik di bawah ini menunjukkan semua karyawan yang dipetakan ke dalam 2 dimensi menggunakan PCA. Setiap warna mewakili segmen/cluster yang berbeda. **Posisi karyawan yang Anda masukkan ditandai dengan bintang hitam.**")

    fig = px.scatter(
        df_cluster_data,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=f'Visualisasi Cluster dengan Model {selected_model_name}',
        hover_data=['MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany'],
        category_orders={"Cluster": sorted(df_cluster_data['Cluster'].unique())} # Urutkan legenda
    )
    
    # Menambahkan tanda untuk input pengguna
    fig.add_scatter(
        x=[input_pca[0, 0]],
        y=[input_pca[0, 1]],
        mode='markers',
        marker=dict(color='black', size=15, symbol='star'),
        name='Posisi Anda'
    )
    
    fig.update_layout(legend_title_text='Cluster')
    st.plotly_chart(fig, use_container_width=True)


# --- Tampilan Laporan Kinerja di Bagian Bawah ---
st.write("---")
st.header("Laporan & Profil Umum Cluster")
st.write(f"Berikut adalah laporan untuk model **{selected_model_name}**.")

model_report_display = report.get(selected_model_name, {})

col1, col2, col3 = st.columns(3)
col1.metric("Silhouette Score", f"{model_report_display.get('silhouette_score', 0):.3f}")
col2.metric("Jumlah Cluster Ditemukan", model_report_display.get('n_clusters_found', 'N/A'))
col3.metric("Jumlah Noise Points", model_report_display.get('n_noise_points', 'N/A'))

st.subheader("Profil Rata-rata Semua Cluster")
cluster_profiles_display = model_report_display.get('cluster_profiles', {})
if cluster_profiles_display:
    profile_all_df = pd.DataFrame.from_dict({int(k): v for k, v in cluster_profiles_display.items()}, orient='index').sort_index()
    profile_all_df.index.name = "Cluster"
    st.dataframe(profile_all_df)
else:
    st.warning("Tidak ada data profil cluster yang ditemukan.")
