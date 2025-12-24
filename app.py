import streamlit as st  #  antarmuka website dashboard
import pandas as pd     # Untuk membaca file
import matplotlib.pyplot as plt # Untuk membuat grafik visualisasi dasar
import seaborn as sns   # Untuk membuat grafik statistik
import io               # Mengelola file di memori
import numpy as np      # Untuk operasi matematika numerik dan array
import time             # Untuk simulasi loading

from sklearn.model_selection import train_test_split # Membagi dataset menjadi data latih (train) dan uji (test)
from sklearn.preprocessing import StandardScaler     # Menyamakan skala data agar performa SVM optimal
from sklearn.svm import SVC                          # Algoritma Support Vector Classifier yang digunakan
from sklearn.decomposition import PCA                # Meringkas dimensi data untuk keperluan visualisasi 2D

# --- Pustaka Evaluasi Model ---
from sklearn.metrics import (
    accuracy_score,         # Menghitung persentase ketepatan prediksi model
    confusion_matrix,       # Matriks untuk melihat detail kesalahan prediksi (Positif/Negatif)
    classification_report,  # Laporan lengkap mencakup Presisi, Recall, dan F1-Score
    roc_curve,              # Menghitung kurva ROC untuk analisis performa klasifikasi
    auc,                    # Skor Area Under Curve (kualitas model secara keseluruhan)
    precision_score,        # Mengukur seberapa akurat saat memprediksi kelas positif
    recall_score            # Mengukur seberapa banyak data positif asli yang berhasil ditebak
)

from sklearn.inspection import permutation_importance # Menghitung fitur mana yang paling berpengaruh terhadap hasil

# ===============================
# 1. KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Dashboard SVM Diabetes Lengkap",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    div.block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

FILE_DEFAULT = "data_diabetes.xlsx" 

# ===============================
# 2. FUNGSI LOAD DATA
# ===============================
@st.cache_data                      # Cache agar tidak load ulang terus
def load_data(uploaded_file):
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=";")
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        return None

# ===============================
# 3. SIDEBAR
# ===============================
st.sidebar.markdown("<h2 style='text-align:center;'>🩺 Kelompok 5 Data Science</h2>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload Data", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    sumber_data = "Data User"
else:
    try:
        df = pd.read_excel(FILE_DEFAULT) 
        sumber_data = "Data Diabetes"
    except:
        df = None
        sumber_data = "Kosong"

if df is not None:
    st.session_state['data'] = df
    st.sidebar.success(f"✅ Sumber: {sumber_data}")
    st.sidebar.caption(f"Rows: {df.shape[0]} | Cols: {df.shape[1]}")
else:
    st.sidebar.warning(f"⚠️ File '{FILE_DEFAULT}' tidak ditemukan.")

menu = st.sidebar.radio(
    "Tahapan Analisis",
    ["📘 Business Understanding", "📊 Data Understanding", "🧹 Data Preparation", "🤖 Modeling", "📈 Evaluasi"]
)






# ===============================
# A. BUSINESS UNDERSTANDING
# ===============================
if menu == "📘 Business Understanding":
    st.title("📘 Business Understanding & Project Overview")
    st.header("1. Latar Belakang & Urgensi Masalah")
    st.markdown("""
    **Diabetes Mellitus** adalah salah satu penyakit kronis paling mematikan di dunia yang terjadi ketika pankreas tidak mampu memproduksi insulin yang cukup, atau ketika tubuh tidak dapat menggunakan insulin yang dihasilkannya secara efektif. 
    
    Tantangan utama dalam dunia medis saat ini adalah **keterlambatan diagnosis**. Banyak pasien baru menyadari kondisi mereka setelah komplikasi terjadi. Oleh karena itu, pendekatan berbasis data (*Data-Driven*) menggunakan **Machine Learning** diperlukan untuk membantu tenaga medis melakukan skrining awal secara cepat, murah, dan akurat.
    """)

    st.divider()

    st.header("2. Pemahaman Data (Dataset Knowledge)")
    st.markdown("""
    Data yang digunakan dalam simulasi ini mengacu pada parameter diagnostik standar untuk diabetes. 
    Setiap fitur memiliki relevansi medis yang kuat terhadap indikasi penyakit:
    """)

    with st.expander("Klik untuk melihat Detail Variabel Data"):
        st.markdown("""
        * **Pregnancies:** Jumlah kehamilan (Riwayat diabetes gestasional meningkatkan risiko).
        * **Glucose:** Konsentrasi glukosa plasma 2 jam (Indikator utama gula darah).
        * **BloodPressure:** Tekanan darah diastolik (Hipertensi sering berkorelasi dengan diabetes).
        * **SkinThickness:** Ketebalan lipatan kulit triceps (Indikator lemak tubuh subkutan).
        * **Insulin:** Kadar insulin serum 2 jam (Menunjukkan resistensi insulin).
        * **BMI (Body Mass Index):** Indikator obesitas, faktor risiko utama diabetes tipe 2.
        * **DiabetesPedigreeFunction:** Skor fungsi genetik (Riwayat penyakit dalam keluarga).
        * **Age:** Usia (Risiko diabetes meningkat seiring bertambahnya usia).
        * **Outcome:** Variabel target (0 = Sehat/Negatif, 1 = Diabetes/Positif).
        """)

    st.divider()
    st.header("3. Metodologi Penelitian (CRISP-DM)")
    st.write("Penelitian ini mengikuti standar industri **Cross-Industry Standard Process for Data Mining (CRISP-DM)**:")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("📊 Data Understanding")
        st.caption("Eksplorasi statistik, distribusi data (Histogram/Violin), dan korelasi antar fitur.")
    with c2:
        st.subheader("🧹 Data Preparation")
        st.caption("Pembersihan data: Handling Missing Value, IQR Capping (Outlier), dan Normalisasi (StandardScaler).")
    with c3:
        st.subheader("🤖 Modeling (SVM)")
        st.caption("Pelatihan model SVM dengan kernel trick (RBF/Linear) untuk memisahkan kelas data.")
    with c4:
        st.subheader("📈 Evaluation")
        st.caption("Pengujian performa menggunakan Confusion Matrix, Accuracy, Precision, Recall, dan ROC Curve.")





# ===============================
# B. DATA UNDERSTANDING
# ===============================
elif menu == "📊 Data Understanding":
    st.title("📊 Data Understanding")

    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📄 Data & Statistik", "📊 Visualisasi Distribusi", "💡 Insight Hubungan"])

        with tab1:
            st.subheader("1. Tinjauan Dataset")
            tampil_mode = st.radio(
                "Pilih Mode Tampilan Angka:",
                ["🔠 Data Asli", "🔢 Data Diselaraskan"],
                horizontal=True,
            )

            if tampil_mode == "🔠 Data Asli":
                df_display = df.copy()
                keterangan = "Menampilkan data dengan nilai asli sesuai dataset."
            else:
                df_display = df.copy()
                numeric_cols = df.select_dtypes(include=np.number).columns
                df_display[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
            st.dataframe(df_display)
            st.divider()

            st.subheader("2. Informasi Dataset")
            c1, c2 = st.columns(2)
            c1.metric("Jumlah Baris", df.shape[0])
            c2.metric("Jumlah Kolom", df.shape[1])

            buffer = io.StringIO()
            df.info(buf=buffer)
            with st.expander("Lihat Detail Tipe Data (Info)"):
                st.text(buffer.getvalue())

            st.subheader("3. Nilai Unik")
            unique_df = pd.DataFrame({
                "Nama Kolom": df.columns,
                "Tipe Data": df.dtypes.astype(str),
                "Jumlah Unik": [df[col].nunique() for col in df.columns],
                "Missing Values": df.isnull().sum()
            })
            st.dataframe(unique_df, use_container_width=True)

            st.divider()

            # --- TAMPILAN 3: STATISTIK DESKRIPTIF ---
            st.subheader("4. Statistik Deskriptif")
            st.write(f"**Ringkasan Statistik ({tampil_mode})**")
            
            # Ini juga otomatis berubah karena menggunakan 'df_display'
            st.dataframe(df_display.describe())

        # --- TAB 2: DISTRIBUSI (LENGKAP) ---
        with tab2:
            numeric_cols = df.select_dtypes(include=np.number).columns
            
            # --- Bagian 1: Pie Chart Target ---
            st.subheader("A. Proporsi Data Target")
            possible_target = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]
            
            col_pie1, col_pie2 = st.columns([1, 2])
            with col_pie1:
                st.write(f"Target: **{possible_target}**")
                st.dataframe(df[possible_target].value_counts())
            with col_pie2:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 3))
                ax_pie.pie(df[possible_target].value_counts(), autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
                ax_pie.set_title(f"Persentase {possible_target}")
                st.pyplot(fig_pie)

            st.divider()

            # --- Bagian 2: Tiga Visualisasi Sekaligus ---
            st.subheader("B. Analisis Distribusi Fitur")
            
            sel_col = st.selectbox("Pilih Kolom Numerik:", numeric_cols)
            
            # GAMBAR 1: HISTOGRAM
            st.write(f"**1. Histogram (Bentuk Sebaran Data) - {sel_col}**")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df[sel_col], kde=True, ax=ax, color="blue")
            ax.set_xlabel(sel_col)
            st.pyplot(fig)
            c_viz1, c_viz2 = st.columns(2)
            
            with c_viz1:
                st.write(f"**2. Boxplot (Cek Outlier) - {sel_col}**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                # Boxplot berwarna oranye agar kontras
                sns.boxplot(x=df[sel_col], ax=ax2, color="orange")
                ax2.set_title("Titik hitam di luar kotak adalah Outlier")
                st.pyplot(fig2)
                
            with c_viz2:
                st.write(f"**3. Violin Plot (Kepadatan) - {sel_col}**")
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                # Violin plot berwarna cyan
                sns.violinplot(x=df[sel_col], ax=ax3, color="cyan")
                ax3.set_title("Bagian gembung menunjukkan data menumpuk")
                st.pyplot(fig3)

        # --- TAB 3: INSIGHT ---
        with tab3:
            st.subheader("1. Heatmap Korelasi")
            if len(numeric_cols) > 1:
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
                st.pyplot(fig4)
            else:
                st.warning("Data numerik kurang.")
            
            st.divider()

            st.subheader("2. Perbandingan Rata-rata per Kelas")
            try:
                mean_df = df.groupby(possible_target)[numeric_cols].mean()
                st.write("**Tabel Rata-rata:**")
                st.dataframe(mean_df.style.highlight_max(axis=0, color='lightcoral')) 
                
                st.write("**Grafik Perbandingan:**")
                comp_col = st.selectbox("Bandingkan Fitur:", numeric_cols, index=0)
                
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                sns.barplot(x=df[possible_target], y=df[comp_col], palette="viridis", ax=ax_bar)
                ax_bar.set_title(f"Rata-rata {comp_col} by {possible_target}")
                st.pyplot(fig_bar)
            except:
                st.warning("Gagal membuat perbandingan grup.")

    else:
        st.error("Data belum dimuat. Silakan upload data di sidebar.")





# ===============================
# C. DATA PREPARATION / PREPROCESSING
# ===============================
elif menu == "🧹 Data Preparation":
    st.title("🧹 Data Preparation")

    if "data" not in st.session_state:
        st.warning("⚠️ Data belum ada. Silakan upload di Data Understanding.")
    else:
        # Ambil data dari session state
        data = st.session_state["data"].copy()
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Tabs Alur Proses
        tab1, tab2, tab3, tab4 = st.tabs([
            "1️⃣ Cleaning Data", 
            "2️⃣ Handling Outlier", 
            "3️⃣ Normalisasi Data", 
            "4️⃣ Finalisasi & Simpan"
        ])

        # --- TAB 1: CLEANING ---
        with tab1:
            st.subheader("Pembersihan Data Dasar")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Missing Values", data.isnull().sum().sum())
            c2.metric("Duplikat", data.duplicated().sum())
            non_num = len(data.columns) - len(numeric_cols)
            c3.metric("Kolom Non-Numerik", non_num, help="Kolom teks otomatis dibuang nanti.")
            
            if data.isnull().sum().sum() > 0:
                st.warning("Mengisi data kosong dengan rata-rata (Mean Imputation).")
                data = data.fillna(data.mean(numeric_only=True))
                st.success("✅ Missing values berhasil ditangani.")
            else:
                st.success("✅ Data bersih dari Missing Values.")

        # --- TAB 2: OUTLIER ---
        with tab2:
            st.subheader("Penanganan Outlier (IQR)")
            
            # Hitung Outlier Awal
            outlier_before = {}
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_before[col] = ((data[col] < lower) | (data[col] > upper)).sum()

            # Proses Capping
            capped_data = data.copy()
            for col in numeric_cols:
                Q1 = capped_data[col].quantile(0.25)
                Q3 = capped_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                capped_data[col] = np.where(capped_data[col] < lower, lower, 
                                   np.where(capped_data[col] > upper, upper, capped_data[col]))

            # Tabel Perbandingan
            st.write("**Ringkasan Jumlah Outlier:**")
            comp_outlier = pd.DataFrame({
                "Fitur": outlier_before.keys(),
                "Sebelum": outlier_before.values(),
                "Sesudah": [0] * len(outlier_before)
            })
            st.dataframe(comp_outlier.set_index("Fitur").T)

            # Visualisasi
            sel_viz = st.selectbox("Cek Visualisasi Outlier:", numeric_cols)
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                st.caption("🔴 Sebelum")
                figA, axA = plt.subplots(figsize=(6, 3))
                sns.boxplot(x=data[sel_viz], color="salmon", ax=axA)
                st.pyplot(figA)
            with c_g2:
                st.caption("🟢 Sesudah")
                figB, axB = plt.subplots(figsize=(6, 3))
                sns.boxplot(x=capped_data[sel_viz], color="lightgreen", ax=axB)
                st.pyplot(figB)

        # --- TAB 3: NORMALISASI ---
        with tab3:
            st.subheader("Simulasi Normalisasi (Standard Scaler)")
            st.markdown("Menyamakan skala data agar rentang nilainya seimbang (Mean=0, Std=1).")
            
            # Demo Normalisasi
            scaler_demo = StandardScaler()
            data_scaled_demo = pd.DataFrame(scaler_demo.fit_transform(capped_data[numeric_cols]), 
                                          columns=numeric_cols)
            
            col_norm = st.selectbox("Pilih Fitur:", numeric_cols, index=1)
            c_n1, c_n2 = st.columns(2)
            with c_n1:
                st.write(f"**Data Asli ({col_norm})**")
                st.caption(f"Range: {capped_data[col_norm].min():.1f} s/d {capped_data[col_norm].max():.1f}")
                fig_raw, ax_raw = plt.subplots(figsize=(6,3))
                sns.histplot(capped_data[col_norm], kde=True, color="orange", ax=ax_raw)
                st.pyplot(fig_raw)
            with c_n2:
                st.write(f"**Setelah Normalisasi**")
                st.caption(f"Range: {data_scaled_demo[col_norm].min():.1f} s/d {data_scaled_demo[col_norm].max():.1f}")
                fig_scl, ax_scl = plt.subplots(figsize=(6,3))
                sns.histplot(data_scaled_demo[col_norm], kde=True, color="purple", ax=ax_scl)
                st.pyplot(fig_scl)
            
            st.success("✅ Skala data berhasil diseragamkan tanpa mengubah pola distribusi.")

        # --- TAB 4: FINALISASI ---
        with tab4:
            st.subheader("Simpan Data untuk Modeling")
            
            all_cols = list(capped_data.columns)
            def_idx = all_cols.index('Outcome') if 'Outcome' in all_cols else len(all_cols) - 1
            target_col = st.selectbox("Label Target (Y):", all_cols, index=def_idx)

            if st.button("💾 Simpan Data & Lanjut Modeling"):
                # 1. Pisahkan X dan y
                X_temp = capped_data.drop(columns=[target_col])
                X = X_temp.select_dtypes(include=[np.number])
                y = capped_data[target_col]

                # 2. Encode Target jika teks
                if y.dtype == 'object':
                    try: y = y.astype('category').cat.codes
                    except: pass
                else:
                    try: y = y.astype(int)
                    except: pass

                # 3. Simpan ke Session
                st.session_state['X'] = X
                st.session_state['y'] = y
                st.session_state['clean_data'] = capped_data
                
                # 4. Notifikasi
                st.success("✅ Data Preparation Selesai!")
                st.write(f"Data siap digunakan di menu Modeling. (Fitur: {X.shape[1]} kolom)")





# ===============================
# D. MODELING (VERSI SIMPLE & BERSIH)
# ===============================
elif menu == "🤖 Modeling":
    st.title("🤖 Modeling SVM")

    # 1. Cek Kesiapan Data
    if 'X' not in st.session_state or 'y' not in st.session_state:
        st.error("⚠️ Data belum siap. Harap selesaikan tahap 'Preprocessing' terlebih dahulu.")
        st.stop()
    else:
        X = st.session_state['X']
        y = st.session_state['y']

        # 2. Konfigurasi Model (Sederhana)
        with st.expander("⚙️ Konfigurasi Model", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                test_size = st.slider("Rasio Data Test (%)", 10, 90, 20, help="Semakin besar, semakin sedikit data untuk belajar.") / 100
                st.caption(f"Train: {100 - (test_size*100):.0f}% | Test: {test_size*100:.0f}%")
            
            with c2:
                kernel = st.selectbox("Jenis Kernel", ["linear", "rbf", "poly", "sigmoid"], index=1,
                                      help="Pilih 'rbf' untuk data kompleks, atau 'linear' untuk data sederhana.")

        # 3. Tombol Training
        if st.button("🚀 Mulai Training Model", use_container_width=True):
            
            # Progress Bar (Fitur Kosmetik)
            progress_text = "Sedang melatih model SVM..."
            my_bar = st.progress(0, text=progress_text)

            try:
                # STEP A: Split Data
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                except ValueError:
                    st.warning("Stratify otomatis dimatikan (Data target tidak seimbang).")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                
                my_bar.progress(30, text="Normalisasi Data...")

                # STEP B: Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                my_bar.progress(60, text="Fitting Algoritma SVM...")

                # STEP C: Training Model (Tanpa C & Gamma manual)
                model = SVC(kernel=kernel, probability=True, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Prediksi
                y_pred = model.predict(X_test_scaled)
                
                my_bar.progress(100, text="Selesai!")
                time.sleep(0.5) 
                my_bar.empty()

                # STEP D: Simpan Hasil
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['X_test_scaled'] = X_test_scaled 
                st.session_state['X_test_raw'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['X_train_raw'] = X_train
                st.session_state['y_train'] = y_train

                # 4. Tampilkan Hasil
                st.success("✅ Model Berhasil Dilatih!")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Akurasi Model", f"{acc*100:.2f}%")
                
                with col_res2:
                    st.info(f"**Info Model:**\nKernel: `{kernel}`\nTest Size: `{test_size*100:.0f}%`")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        # 5. Monitoring & Visualisasi
        if 'model' in st.session_state:
            st.divider()
            
            tab_info, tab_viz = st.tabs(["📄 Data Split", "📊 Visualisasi 2D (PCA)"])
            
            with tab_info:
                X_tr = st.session_state['X_train_raw']
                X_ts = st.session_state['X_test_raw']
                
                c1, c2 = st.columns(2)
                c1.markdown(f"**Data Latih (Train)**\nJumlah: `{X_tr.shape[0]}` baris")
                c2.markdown(f"**Data Uji (Test)**\nJumlah: `{X_ts.shape[0]}` baris")
                
                with st.expander("Lihat Sampel Data Training"):
                    st.dataframe(X_tr.head())

            with tab_viz:
                st.markdown("**Proyeksi Data 2 Dimensi (PCA)**")
                st.caption("Visualisasi ini meringkas banyak fitur menjadi 2 dimensi untuk melihat pola sebaran data.")
                
                try:
                    scaler_viz = st.session_state['scaler']
                    X_viz_scaled = scaler_viz.transform(st.session_state['X'])
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_viz_scaled)
                    
                    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
                    pca_df['Target'] = st.session_state['y'].values
                    
                    fig_pca, ax_pca = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Target', palette='viridis', ax=ax_pca)
                    ax_pca.set_title("Sebaran Data (PCA)")
                    st.pyplot(fig_pca)
                    
                except Exception as e:
                    st.warning(f"Gagal menampilkan PCA: {e}")





# ===============================
# E. EVALUASI & PREDIKSI (FIX DETEKSI OTOMATIS)
# ===============================
elif menu == "📈 Evaluasi":
    st.title("📈 Evaluasi & Simulasi Prediksi")

    if "model" not in st.session_state:
        st.warning("⚠️ Belum ada model. Silakan lakukan Training dulu di menu '4. Pembuatan Model'.")
    else:
        # Ambil data
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        model = st.session_state["model"]
        X_test_scaled = st.session_state["X_test_scaled"]
        X_test_raw = st.session_state["X_test_raw"]

        # --- 4 TAB UTAMA ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Performa Model", 
            "🟦 Confusion Matrix", 
            "📈 Kurva ROC", 
            "🤖 Simulasi Cek Kesehatan"
        ])

        # === TAB 1: PERFORMA ===
        with tab1:
            st.subheader("Skor Kualitas Model")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            c1, c2 = st.columns(2)
            c1.metric("Akurasi", f"{acc*100:.2f}%")
            c2.metric("Presisi", f"{prec*100:.2f}%")

        # === TAB 2: CONFUSION MATRIX ===
        with tab2:
            st.subheader("Matriks Kebingungan")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            st.pyplot(fig_cm)

        # === TAB 3: KURVA ROC ===
        with tab3:
            st.subheader("Kurva ROC")
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                    ax_roc.plot(fpr, tpr, color='orange', lw=2, label=f'AUC = {roc_auc:.2f}')
                    ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
                    ax_roc.legend()
                    st.pyplot(fig_roc)
                except:
                    st.info("Kurva tidak tersedia.")
            else:
                st.info("Model tidak mendukung ROC.")

        # === TAB 4: SIMULASI (LOGIKA DETEKSI KATA KUNCI) ===
        with tab4:
            st.subheader("Simulasi Diagnosa Mandiri")
            st.info("Masukkan data klinis pasien di bawah ini.")

            with st.form(key='form_simulasi_smart'):
                input_data = {}
                cols = st.columns(3) 
                
                # Kita looping kolom, tapi cek isinya pakai "Kata Kunci"
                # Biar mau bahasa Indo/Inggris tetap terdeteksi
                for i, col in enumerate(X_test_raw.columns):
                    col_lower = col.lower() # Ubah nama kolom jadi huruf kecil semua biar gampang dicek
                    
                    with cols[i % 3]:
                        
                        # --- 1. LOGIKA GENETIK (Cari kata 'genetik' atau 'pedigree') ---
                        # Ini yang membuat jadi RADIO BUTTON
                        if 'genetik' in col_lower or 'pedigree' in col_lower:
                            # Pilihan Radio Button 0 atau 1
                            pilihan = st.radio(
                                f"**{col}**", 
                                [0, 1], 
                                index=0,
                                format_func=lambda x: "0 (Tidak Ada)" if x == 0 else "1 (Ada Riwayat)",
                                help="Pilih 1 jika ada riwayat keluarga."
                            )
                            input_data[col] = float(pilihan)

                        # --- 2. LOGIKA UMUR & HAMIL (Cari kata 'age', 'usia', 'umur', 'hamil', 'pregnan') ---
                        # Ini membuat jadi ANGKA BULAT (Tidak ada koma 3,50)
                        elif any(x in col_lower for x in ['age', 'usia', 'umur', 'hamil', 'pregnan']):
                            mean_val = int(X_test_raw[col].mean())
                            input_data[col] = st.number_input(
                                label=f"**{col}**",
                                min_value=0,
                                step=1,        # Wajib loncat 1 (bulat)
                                value=mean_val,
                                format="%d"    # Format Integer (tanpa koma)
                            )
                            
                        # --- 3. STANDAR (Gula, BMI, Tensi, dll) ---
                        else:
                            mean_val = float(X_test_raw[col].mean())
                            input_data[col] = st.number_input(
                                label=f"**{col}**",
                                min_value=0.0,
                                value=mean_val,
                                format="%.2f"  # Boleh koma
                            )
                        
                        st.write("") 

                st.markdown("---")
                tombol_prediksi = st.form_submit_button("🔍 Analisa Risiko Sekarang", use_container_width=True)

            # Logika Prediksi
            if tombol_prediksi:
                scaler = st.session_state['scaler']
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                
                hasil_kelas = model.predict(input_scaled)[0]
                
                st.divider()
                if hasil_kelas == 1:
                    st.error("🚨 **HASIL PREDIKSI: BERISIKO (POSITIF)**")
                    st.write("Sistem mendeteksi indikasi diabetes. Disarankan konsultasi dokter.")
                else:
                    st.success("✅ **HASIL PREDIKSI: SEHAT (NEGATIF)**")
                    st.write("Sistem memprediksi kondisi aman.")