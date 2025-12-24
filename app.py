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
# A. BUSINESS UNDERSTANDING (VERSI LENGKAP & AKADEMIS)
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
# E. EVALUASI MODEL (LENGKAP & DINAMIS)
# ===============================
elif menu == "📈 Evaluasi":
    st.title("📈 Evaluasi Model")

    if "model" not in st.session_state:
        st.warning("⚠️ Belum ada model. Silakan lakukan Training di menu Modeling terlebih dahulu.")
    else:
        # Ambil semua data dari session
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        model = st.session_state["model"]
        X_test_scaled = st.session_state["X_test_scaled"]
        X_test_raw = st.session_state["X_test_raw"]

        # --- 1. METRIK PERFORMA ---
        st.subheader("1. Performa Model")
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Akurasi", f"{acc*100:.2f}%", help="Persentase total tebakan benar")
        k2.metric("Presisi", f"{prec*100:.2f}%", help="Ketepatan menebak positif")

        st.divider()

        # --- 2. VISUALISASI EVALUASI ---
        eval_tab1, eval_tab2, eval_tab3 = st.tabs(["📊 Confusion Matrix", "📉 ROC Curve", "⭐ Feature Importance"])

        with eval_tab1:
            st.write("**Confusion Matrix** (Detail Benar vs Salah)")
            cm = confusion_matrix(y_test, y_pred)
            
            c_cm1, c_cm2 = st.columns([1, 2])
            with c_cm1:
                st.info("Diagonal biru tua menunjukkan jumlah prediksi yang BENAR.")
            with c_cm2:
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_ylabel("Data Aktual")
                ax_cm.set_xlabel("Prediksi Model")
                st.pyplot(fig_cm)
            
            st.write("**Classification Report:**")
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report_dict).transpose().style.background_gradient(cmap='Blues'))

        with eval_tab2:
            st.write("**ROC Curve & AUC**")
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax_roc.set_xlim([0.0, 1.0])
                    ax_roc.set_ylim([0.0, 1.05])
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.set_title('Receiver Operating Characteristic')
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc)
                    st.success(f"Nilai AUC: {roc_auc:.2f} (Semakin dekat 1.0 semakin bagus)")
                except:
                    st.warning("Gagal membuat ROC Curve (Mungkin target bukan biner/multiclass).")
            else:
                st.info("Model ini tidak mendukung probabilitas untuk kurva ROC.")

        with eval_tab3:
            st.write("**Feature Importance (Permutation Method)**")
            st.caption("Fitur mana yang paling mempengaruhi keputusan model?")
            
            with st.spinner("Menghitung tingkat kepentingan fitur..."):
                results = permutation_importance(model, X_test_scaled, y_test, scoring='accuracy')
                importance = results.importances_mean
                
                imp_df = pd.DataFrame({
                    'Fitur': X_test_raw.columns,
                    'Importance': importance
                }).sort_values(by='Importance', ascending=False)
                
                fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                sns.barplot(data=imp_df, x='Importance', y='Fitur', palette='viridis', ax=ax_imp)
                ax_imp.set_title("Faktor Paling Berpengaruh")
                st.pyplot(fig_imp)

        # --- 3. SIMULASI PREDIKSI ---
        st.divider()
        st.subheader("🤖 Simulasi Prediksi Manual")
        col_list = X_test_raw.columns.tolist()
        is_diabetes_context = 'Glucose' in col_list and 'BMI' in col_list

        if is_diabetes_context:
            st.info("💡 Mode Diabetes Terdeteksi: Form menggunakan referensi medis.")
            notes_dict = {
                "Pregnancies": "Normal: 0-17", "Glucose": "Normal: <140 | Diabetes: >200",
                "BloodPressure": "Normal: <80", "SkinThickness": "Range: 10-50mm",
                "Insulin": "Normal: 2.6-24.9", "BMI": "Normal: 18.5-24.9",
                "DiabetesPedigreeFunction": "Skor genetik", "Age": "Usia (Tahun)"
            }
        else:
            st.info(f"💡 Mode Generik: Prediksi berdasarkan dataset '{list(X_test_raw.columns[:3])}...'")
            notes_dict = {}

        # Form Input Dinamis
        input_data = {}
        cols = st.columns(3)
        for i, col in enumerate(X_test_raw.columns):
            with cols[i % 3]:
                min_v = float(X_test_raw[col].min())
                max_v = float(X_test_raw[col].max())
                mean_v = float(X_test_raw[col].mean())
                if is_diabetes_context:
                    help_text = notes_dict.get(col, f"Range: {min_v:.1f} - {max_v:.1f}")
                else:
                    help_text = f"Range data valid: {min_v:.1f} s/d {max_v:.1f}"

                input_data[col] = st.number_input(
                    label=f"**{col}**",
                    min_value=0.0,
                    value=mean_v,
                    format="%.2f",
                    help=help_text
                )

                st.caption(f"ℹ️ {help_text}")
                st.write("")

        if st.button("🔍 Cek Hasil Prediksi", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            scaler = st.session_state['scaler']
            input_scaled = scaler.transform(input_df)
            
            res = model.predict(input_scaled)[0]
            
            st.markdown("---")
            if res == 1:
                if is_diabetes_context:
                    st.error("🚨 **HASIL: POSITIF DIABETES**")
                else:
                    st.error("🚨 **HASIL PREDIKSI: KELAS 1 (POSITIF)**")
            else:
                if is_diabetes_context:
                    st.success("✅ **HASIL: NEGATIF**")
                else:
                    st.success("✅ **HASIL PREDIKSI: KELAS 0 (NEGATIF)**")