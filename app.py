import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Dashboard SVM Diabetes",
    layout="wide"
)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.markdown(
    """
    <h2 style='text-align:center;'>🩺 SVM Diabetes</h2>
    <p style='text-align:center;color:gray;'>CRISP-DM Dashboard</p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

menu = st.sidebar.radio(
    "📌 Tahapan Analisis",
    [
        "📘 Business Understanding",
        "📊 Data Understanding",
        "🧹 Preprocessing",
        "🤖 Modeling",
        "📈 Evaluasi"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Tugas Data Science")

# ===============================
# BUSINESS UNDERSTANDING
# ===============================
if menu == "📘 Business Understanding":
    st.title("📘 Business Understanding")

    st.markdown("""
    ### 🎯 Latar Belakang
    Diabetes merupakan penyakit kronis yang membutuhkan deteksi dini.
    Pemanfaatan **Machine Learning** diharapkan mampu membantu proses
    diagnosis secara cepat dan akurat.

    ### 🎯 Tujuan
    - Menganalisis data diabetes
    - Membangun model klasifikasi menggunakan **SVM**
    - Mengevaluasi performa model

    ### 🧭 Metodologi
    Penelitian ini menggunakan pendekatan **CRISP-DM**:
    1. Business Understanding  
    2. Data Understanding  
    3. Preprocessing  
    4. Modeling  
    5. Evaluasi
    """)

# ===============================
# DATA UNDERSTANDING
# ===============================
elif menu == "📊 Data Understanding":
    st.title("📊 Data Understanding")

    file = st.file_uploader("📂 Upload Dataset Diabetes (CSV)", type=["xlsx", "csv"])

    if file is not None:
        data = pd.read_csv(file, sep=";")
        st.session_state["data"] = data

        # ===============================
        # PREVIEW DATA
        # ===============================
        st.subheader("🔍 Preview Dataset")
        st.dataframe(data)

        # ===============================
        # JUMLAH BARIS & KOLOM
        # ===============================
        st.subheader("📐 Ukuran Dataset")
        col1, col2 = st.columns(2)
        col1.metric("Jumlah Baris", data.shape[0])
        col2.metric("Jumlah Kolom", data.shape[1])

        # ===============================
        # INFO DATA
        # ===============================
        st.subheader("ℹ️ Informasi Dataset (data.info())")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

        # ===============================
        # NILAI UNIK
        # ===============================
        st.subheader("🔢 Nilai Unik Setiap Kolom")
        unique_df = pd.DataFrame({
            "Kolom": data.columns,
            "Jumlah Nilai Unik": [data[col].nunique() for col in data.columns]
        })
        st.dataframe(unique_df)

        # ===============================
        # DESKRIPSI DATA
        # ===============================
        st.subheader("📊 Statistik Deskriptif (data.describe())")
        st.dataframe(data.describe())

        # ===============================
        # KORELASI
        # ===============================
        st.subheader("📈 Korelasi Antar Atribut")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    else:
        st.info("Silakan upload dataset untuk melihat Data Understanding.")

# ===============================
# PREPROCESSING
# ===============================
elif menu == "🧹 Preprocessing":
    st.title("🧹 Data Preprocessing")

    if "data" not in st.session_state:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu.")
    else:
        data = st.session_state["data"]
        numeric_cols = data.select_dtypes(include=np.number).columns

        # ===============================
        # MISSING VALUE & DUPLIKAT
        # ===============================
        st.subheader("🧾 Pemeriksaan Kualitas Data")

        col1, col2 = st.columns(2)
        col1.metric("Missing Value", data.isnull().sum().sum())
        col2.metric("Data Duplikat", data.duplicated().sum())

        # ===============================
        # HITUNG OUTLIER (IQR)
        # ===============================
        st.subheader("🚨 Deteksi Outlier (Metode IQR)")

        outlier_count = {}
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count[col] = ((data[col] < lower) | (data[col] > upper)).sum()

        st.dataframe(pd.DataFrame.from_dict(
            outlier_count, orient="index", columns=["Jumlah Outlier"]
        ))

        # ===============================
        # BOXPLOT SEBELUM
        # ===============================
        st.subheader("📦 Boxplot Sebelum Penanganan Outlier")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=data[numeric_cols], ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        st.pyplot(fig1)

        # ===============================
        # PENJELASAN IQR CAPPING
        # ===============================
        st.subheader("🧮 Metode IQR Capping")

        st.markdown("""
        **Langkah perhitungan IQR Capping:**
        1. Hitung Kuartil 1 (Q1) dan Kuartil 3 (Q3)
        2. Hitung IQR = Q3 − Q1
        3. Tentukan batas:
           - Batas bawah = Q1 − 1.5 × IQR  
           - Batas atas = Q3 + 1.5 × IQR
        4. Nilai di luar batas diganti (capping) dengan nilai batas terdekat
        """)

        # ===============================
        # PROSES IQR CAPPING
        # ===============================
        capped_data = data.copy()

        for col in numeric_cols:
            Q1 = capped_data[col].quantile(0.25)
            Q3 = capped_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            capped_data[col] = np.where(
                capped_data[col] < lower, lower,
                np.where(capped_data[col] > upper, upper, capped_data[col])
            )

        st.session_state["clean_data"] = capped_data

        # ===============================
        # OUTLIER SETELAH
        # ===============================
        st.subheader("✅ Outlier Setelah IQR Capping")

        after_outlier = {}
        for col in numeric_cols:
            Q1 = capped_data[col].quantile(0.25)
            Q3 = capped_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            after_outlier[col] = ((capped_data[col] < lower) | (capped_data[col] > upper)).sum()

        st.dataframe(pd.DataFrame.from_dict(
            after_outlier, orient="index", columns=["Jumlah Outlier"]
        ))

        # ===============================
        # BOXPLOT SESUDAH
        # ===============================
        st.subheader("📦 Boxplot Setelah Penanganan Outlier")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=capped_data[numeric_cols], ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        st.pyplot(fig2)

        st.success("✅ Preprocessing selesai & data siap untuk modeling")

# ===============================
# MODELING
# ===============================
elif menu == "🤖 Modeling":
    st.title("🤖 Modeling SVM")

    if "clean_data" not in st.session_state:
        st.warning("⚠️ Lakukan preprocessing terlebih dahulu.")
    else:
        data = st.session_state["clean_data"]

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        st.subheader("⚙️ Parameter Modeling")
        test_size = st.slider("Porsi Data Uji", 0.1, 0.4, 0.2)
        kernel = st.selectbox("Kernel SVM", ["linear", "rbf", "poly"])

        # ===============================
        # SPLIT DATA
        # ===============================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        st.subheader("📐 Hasil Pembagian Data")
        col1, col2 = st.columns(2)
        col1.metric("Data Train", X_train.shape[0])
        col2.metric("Data Test", X_test.shape[0])

        # ===============================
        # TAMPILKAN DATA
        # ===============================
        st.subheader("📄 X_train")
        st.dataframe(X_train.head())

        st.subheader("📄 X_test")
        st.dataframe(X_test.head())

        st.subheader("🎯 y_train")
        st.dataframe(y_train.head())

        st.subheader("🎯 y_test")
        st.dataframe(y_test.head())

        # ===============================
        # SCALING
        # ===============================
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ===============================
        # TRAINING MODEL
        # ===============================
        model = SVC(kernel=kernel)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        st.session_state.update({
        "model": model,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred
        })

        st.success("✅ Model SVM berhasil dibangun")

# ===============================
# EVALUASI
# ===============================
elif menu == "📈 Evaluasi":
    st.title("📈 Evaluasi Model SVM")

    required_keys = ["model", "X_test", "y_test", "clean_data"]

    if not all(k in st.session_state for k in required_keys):
        st.warning("⚠️ Silakan selesaikan tahap Modeling terlebih dahulu.")
    else:
        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        data = st.session_state["clean_data"]

        # ===============================
        # PREDIKSI
        # ===============================
        y_pred = model.predict(X_test)

        # ===============================
        # AKURASI
        # ===============================
        st.subheader("🎯 Akurasi Model")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.4f}")

        # ===============================
        # CONFUSION MATRIX
        # ===============================
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax_cm
        )
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        st.pyplot(fig_cm)

        # ===============================
        # CLASSIFICATION REPORT
        # ===============================
        st.subheader("📄 Classification Report")
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

        # ===============================
        # PAIRPLOT
        # ===============================
        st.subheader("🔍 Pairplot Antar Atribut Utama")

        target_col = data.columns[-1]
        fitur_utama = data.columns[:-1][:4]

        pairplot_data = data[list(fitur_utama) + [target_col]]

        fig_pair = sns.pairplot(
            pairplot_data,
            hue=target_col,
            diag_kind="kde"
        )

        st.pyplot(fig_pair)
