import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_curve, auc, precision_score, recall_score
)
from sklearn.inspection import permutation_importance

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
@st.cache_data
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
    ["📘 Business Understanding", "📊 Data Understanding", "🧹 Preprocessing", "🤖 Modeling", "📈 Evaluasi"]
)

# ===============================
# A. BUSINESS UNDERSTANDING
# ===============================
if menu == "📘 Business Understanding":
    st.title("📘 Business Understanding")
    st.markdown("""
    ### 🎯 Latar Belakang
    Diabetes merupakan penyakit kronis yang membutuhkan deteksi dini. Dashboard ini menggunakan 
    **Support Vector Machine (SVM)** untuk klasifikasi pasien.
    
    ### 🎯 Tujuan
    1. Menganalisis karakteristik data pasien.
    2. Membersihkan data dari noise dan outlier.
    3. Membangun model prediksi yang akurat.
    
    ### 🧭 Alur CRISP-DM:
    * **Data Understanding:** Statistik & Visualisasi distribusi.
    * **Preprocessing:** IQR Capping & Normalisasi.
    * **Modeling:** Training SVM dengan kernel trick.
    * **Evaluasi:** Confusion Matrix, ROC, & Pairplot.
    """)
    if df is not None:
        st.success("✅ Data siap! Silakan lanjut ke menu berikutnya.")

# ===============================
# B. DATA UNDERSTANDING
elif menu == "📊 Data Understanding":
    st.title("📊 Data Understanding")

    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📄 Statistik Detail", "📊 Visualisasi Distribusi", "🔥 Korelasi"])

        with tab1:
            st.subheader("1. Cuplikan Data")
            st.dataframe(df.head())

            st.subheader("2. Informasi Dataset")
            col1, col2 = st.columns(2)
            col1.metric("Jumlah Baris", df.shape[0])
            col2.metric("Jumlah Kolom", df.shape[1])

            buffer = io.StringIO()
            df.info(buf=buffer)
            with st.expander("Detail Info"):
                st.text(buffer.getvalue())

            st.subheader("3. Nilai Unik & Statistik")
            unique_df = pd.DataFrame({
                "Nama Kolom": df.columns,
                "Tipe Data": df.dtypes.astype(str),
                "Jumlah Unik": [df[col].nunique() for col in df.columns],
                "Missing Values": df.isnull().sum()
            })
            st.dataframe(unique_df, use_container_width=True)

            st.subheader("4. Statistik Deskriptif")
            st.dataframe(df.describe())

        with tab2:
            st.subheader("Distribusi Data")
            numeric_cols = df.select_dtypes(include=np.number).columns
            sel_col = st.selectbox("Pilih Kolom:", numeric_cols)
            
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots()
                sns.histplot(df[sel_col], kde=True, ax=ax, color="red")
                ax.set_title(f"Histogram {sel_col}")
                st.pyplot(fig)
            with c2:
                fig2, ax2 = plt.subplots()
                sns.boxplot(x=df[sel_col], ax=ax2, color="cyan")
                ax2.set_title(f"Boxplot {sel_col}")
                st.pyplot(fig2)
            
            st.divider()
            # Pie Chart Target
            st.subheader("Proporsi Target")
            # Coba cari kolom target otomatis
            possible_target = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]
            target_counts = df[possible_target].value_counts()
            
            fig3, ax3 = plt.subplots(figsize=(4,4))
            ax3.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
            st.pyplot(fig3)

            st.write(f"**Violin Plot {sel_col}**")
            fig2, ax2 = plt.subplots()
            sns.violinplot(y=df[sel_col], color="cyan", ax=ax2)
            ax2.set_title(f"Kepadatan Data {sel_col}")
            st.pyplot(fig2)

        with tab3:
            st.subheader("Heatmap Korelasi")
            if len(numeric_cols) > 1:
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
                st.pyplot(fig4)
            else:
                st.warning("Data numerik kurang.")
    else:
        st.error("Data belum dimuat.")

# ===============================
# C. PREPROCESSING
# ===============================
elif menu == "🧹 Preprocessing":
    st.title("🧹 Data Preprocessing")

    if "data" not in st.session_state:
        st.warning("⚠️ Data belum ada. Silakan upload di Data Understanding.")
    else:
        data = st.session_state["data"].copy()
        all_cols = data.columns
        numeric_cols = data.select_dtypes(include=np.number).columns
        non_numeric_cols = list(set(all_cols) - set(numeric_cols))

        st.subheader("1. Cek Kualitas Data")
        c1, c2, c3 = st.columns(3)
        c1.metric("Missing Value", data.isnull().sum().sum())
        c2.metric("Duplikat", data.duplicated().sum())
        c3.metric("Kolom Non-Angka", len(non_numeric_cols))

        if len(non_numeric_cols) > 0:
            st.warning(f"⚠️ Ditemukan kolom Teks yang tidak bisa diproses SVM: {non_numeric_cols}")
            st.info("Sistem akan otomatis mengabaikan kolom tersebut saat modeling.")

        st.divider()
        st.subheader("2. Penanganan Outlier (IQR Capping)")

        outlier_before = {}
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_before[col] = ((data[col] < (Q1 - 1.5*IQR)) | (data[col] > (Q3 + 1.5*IQR))).sum()
        
        capped_data = data.copy()
        for col in numeric_cols:
            Q1 = capped_data[col].quantile(0.25)
            Q3 = capped_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            capped_data[col] = np.where(capped_data[col] < lower, lower, 
                               np.where(capped_data[col] > upper, upper, capped_data[col]))

        st.write("**Jumlah Outlier Terdeteksi (Sebelum vs Sesudah):**")
        compare_df = pd.DataFrame({
            "Kolom": outlier_before.keys(),
            "Outlier Awal": outlier_before.values(),
            "Outlier Akhir": [0]*len(outlier_before) 
        })
        st.dataframe(compare_df.T)

        st.write("**Visualisasi Perubahan (Before vs After):**")
        col_viz = st.selectbox("Pilih Fitur:", numeric_cols)
        
        c_viz1, c_viz2 = st.columns(2)
        with c_viz1:
            st.caption("🔴 Sebelum")
            figA, axA = plt.subplots(figsize=(6,4))
            sns.boxplot(y=data[col_viz], color="salmon", ax=axA)
            st.pyplot(figA)
        with c_viz2:
            st.caption("🟢 Sesudah")
            figB, axB = plt.subplots(figsize=(6,4))
            sns.boxplot(y=capped_data[col_viz], color="lightgreen", ax=axB)
            st.pyplot(figB)

        st.divider()
        st.subheader("3. Pemisahan Fitur & Target")
        target_col = st.selectbox("Pilih Kolom Target (Label):", all_cols, index=len(all_cols)-1)

        if st.button("Simpan Data & Lanjut Modeling"):
            X_temp = capped_data.drop(columns=[target_col])

            X = X_temp.select_dtypes(include=[np.number])

            y = capped_data[target_col]

            if y.dtype == 'object':
                try:
                    y = y.astype('category').cat.codes
                    st.info(f"Label target '{target_col}' berupa teks, otomatis dikonversi jadi angka.")
                except:
                    pass
            else:
                try: y = y.astype(int)
                except: pass

            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['clean_data'] = capped_data
            
            st.success(f"✅ Data tersimpan! Kolom teks pada fitur (X) otomatis dibuang agar tidak error.")
            st.write(f"Fitur yang dipakai ({X.shape[1]} kolom): {list(X.columns)}")

            # Mengubah dataframe jadi file Excel di memori
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                capped_data.to_excel(writer, index=False, sheet_name='Sheet1')
            processed_data = output.getvalue()

            st.download_button(
                label="📥 Download Data Bersih (Excel)",
                data=processed_data,
                file_name="data_clean_diabetes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ===============================
# D. MODELING
# ===============================
elif menu == "🤖 Modeling":
    st.title("🤖 Modeling SVM")

    if 'X' not in st.session_state:
        st.warning("⚠️ Data belum siap. Silakan ke Preprocessing dulu.")
    else:
        X = st.session_state['X']
        y = st.session_state['y']

        with st.expander("⚙️ Konfigurasi Parameter Model", expanded=True):
            col_p1, col_p2 = st.columns(2)
            with col_p1: 
                test_size = st.slider("Ukuran Data Test (%)", 10, 100, 50) / 100
                st.caption("Semakin besar, data latih semakin sedikit.")
            with col_p2: 
                kernel = st.selectbox("Kernel SVM", ["linear", "rbf", "poly", "sigmoid"])
                st.caption("RBF biasanya bagus untuk data kompleks.")

        if st.button("🚀 Mulai Training Model"):
            with st.spinner("Sedang melatih..."):
                try:
                    # Split Data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                except ValueError:
                    st.warning("Stratify gagal (target kontinu?), lanjut tanpa stratify.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                # Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Training
                model = SVC(kernel=kernel, probability=True) # Probability=True wajib buat ROC
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                # Simpan Semua
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler 
                st.session_state['X_test'] = X_test_scaled
                # Simpan data asli X_test dan y_test untuk ditampilkan
                st.session_state['X_test_raw'] = X_test 
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['X_train_raw'] = X_train
                st.session_state['y_train'] = y_train

                st.success("✅ Training Selesai!")
                st.metric("Akurasi Model", f"{accuracy_score(y_test, y_pred)*100:.2f}%")

        # --- MONITORING DATA SPLIT (FITUR DIMINTA) ---
        if 'model' in st.session_state:
            st.divider()
            st.subheader("🔍 Detail Data Splitting")
            
            # Ambil data dari session state
            X_tr = st.session_state['X_train_raw']
            X_ts = st.session_state['X_test_raw']
            y_tr = st.session_state['y_train']
            y_ts = st.session_state['y_test']

            c1, c2 = st.columns(2)
            c1.info(f"Data Latih (Train): {X_tr.shape[0]} baris")
            c2.info(f"Data Uji (Test): {X_ts.shape[0]} baris")

            with st.expander("📄 Lihat Tabel Data Train & Test"):
                st.write("**X_train (5 Baris Pertama):**")
                st.dataframe(X_tr.head())
                st.write("**y_train (Target Latih):**")
                st.dataframe(y_tr.head())
            
            # --- VISUALISASI PCA (BIAR KEREN) ---
            st.subheader("Visualisasi Sebaran Data (PCA 2D)")
            pca = PCA(n_components=2)
            # Pakai data yang sudah di scale
            scaler_viz = st.session_state['scaler']
            X_viz = scaler_viz.transform(st.session_state['X'])
            X_pca = pca.fit_transform(X_viz)
            
            pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            pca_df['Target'] = st.session_state['y'].values
            
            fig_pca, ax_pca = plt.subplots(figsize=(8,5))
            sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Target', palette='viridis', ax=ax_pca)
            ax_pca.set_title("Proyeksi Data 2D (PCA)")
            st.pyplot(fig_pca)


# ===============================
# E. EVALUASI
# ===============================
elif menu == "📈 Evaluasi":
    st.title("📈 Evaluasi Model")

    if "model" not in st.session_state:
        st.warning("⚠️ Belum ada model. Silakan Training dulu.")
    else:
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        model = st.session_state["model"]
        X_test_scaled = st.session_state["X_test"]
        X_test_raw = st.session_state["X_test_raw"] # Data asli untuk pairplot

        # 1. METRIK UTAMA (Dihitung aman agar tidak error key)
        st.subheader("1. Performa Utama")
        acc = accuracy_score(y_test, y_pred)
        # Average weighted biar aman untuk multi-class atau imbalance
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Akurasi", f"{acc*100:.2f}%")
        c2.metric("Presisi (Weighted)", f"{prec*100:.2f}%")
        c3.metric("Recall (Weighted)", f"{rec*100:.2f}%")

        # 2. DETAIL LAPORAN
        st.divider()
        col_g, col_t = st.columns([1, 1.5])
        
        with col_g:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_ylabel("Aktual")
            ax_cm.set_xlabel("Prediksi")
            st.pyplot(fig_cm)
        
        with col_t:
            st.subheader("Classification Report")
            # Convert report to DataFrame agar AMAN dari error Key
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Blues'))

        # --- MULAI KODE BARU (FEATURE IMPORTANCE) ---
        st.divider()
        st.subheader("2. Faktor Paling Berpengaruh")
        
        # Kita pakai spinner karena hitungannya agak berat dikit
        with st.spinner("Menghitung Feature Importance..."):
            # Hitung tingkat kepentingan fitur
            results = permutation_importance(model, X_test_scaled, y_test, scoring='accuracy')
            importance = results.importances_mean
            
            # Ambil nama kolom dari data X asli
            feature_names = X_test_raw.columns
            
            # Buat DataFrame biar mudah di-plot
            imp_df = pd.DataFrame({
                'Fitur': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            
            # Gambar Grafik Batang
            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            sns.barplot(data=imp_df, x='Importance', y='Fitur', palette='magma', ax=ax_imp)
            ax_imp.set_title("Tingkat Kepentingan Fitur")
            ax_imp.set_xlabel("Nilai Importance")
            st.pyplot(fig_imp)
        # --- SELESAI KODE BARU ---

        # 3. GRAFIK LANJUTAN (ROC & PAIRPLOT)
        st.divider()
        tab_e1, tab_e2 = st.tabs(["📈 ROC Curve", "🔍 Pairplot (Sebaran)"])

        with tab_e1:
            try:
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='orange', lw=2, label=f'AUC = {roc_auc:.2f}')
                ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
                ax_roc.set_title('ROC Curve')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend()
                st.pyplot(fig_roc)
            except:
                st.info("Kernel ini tidak support probabilitas untuk ROC.")

        with tab_e2:
            st.write("Pairplot memvisualisasikan hubungan antar fitur pada data Test.")
            viz_data = X_test_raw.copy()
            viz_data['Target'] = y_test.values

            if viz_data.shape[1] > 6:
                st.warning("Fitur terlalu banyak, menampilkan 5 fitur pertama saja.")
                viz_cols = viz_data.columns[:5].tolist() + ['Target']
                viz_data = viz_data[viz_cols]
            
            fig_pair = sns.pairplot(viz_data, hue='Target', diag_kind='kde')
            st.pyplot(fig_pair)
        
        # 4. SIMULASI PREDIKSI
        st.divider()
        st.subheader("🤖 Simulasi Prediksi Manual")
        
        cols = st.columns(4)
        input_data = {}
        for i, col in enumerate(X_test_raw.columns):
            with cols[i % 4]:
                input_data[col] = st.number_input(col, value=0.0)
        
        if st.button("🔍 Cek Prediksi"):
            input_df = pd.DataFrame([input_data])
            scaler = st.session_state['scaler']
            input_scaled = scaler.transform(input_df)
            res = model.predict(input_scaled)[0]
            
            if res == 1:
                st.error("Hasil: Positif Diabetes")
            else:
                st.success("Hasil: Negatif (Sehat)")