import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# SIDEBAR NAVIGASI
# ===============================
st.sidebar.markdown(
    """
    <h2 style='text-align:center;'>🩺 SVM Diabetes</h2>
    <p style='text-align:center;color:gray;'>Data Science Dashboard</p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

menu = st.sidebar.radio(
    "📌 Navigasi",
    [
        "🏠 Dashboard Utama",
        "🤖 Training & Evaluasi SVM",
        "✍️ Prediksi Manual"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("<small>Dibuat untuk tugas Data Science</small>", unsafe_allow_html=True)

# ===============================
# DASHBOARD UTAMA
# ===============================
if menu == "🏠 Dashboard Utama":
    st.title("🩺 Dashboard Klasifikasi Diabetes")
    st.write("""
    Dashboard ini menggunakan **Support Vector Machine (SVM)**  
    untuk melakukan klasifikasi penyakit diabetes.
    """)

    st.markdown("""
    ### 🎯 Tujuan
    - Melatih model SVM
    - Evaluasi performa model
    - Prediksi diabetes pasien baru

    ### 🧭 Alur
    1️⃣ Upload dataset  
    2️⃣ Training & evaluasi  
    3️⃣ Prediksi manual
    """)

    st.info("Silakan pilih menu **🤖 Training & Evaluasi SVM** untuk memulai.")

# ===============================
# TRAINING & EVALUASI
# ===============================
elif menu == "🤖 Training & Evaluasi SVM":
    st.title("🤖 Training & Evaluasi Model SVM")

    file = st.file_uploader("📂 Upload Dataset Diabetes (CSV)", type=["csv"])

    if file is not None:
        data = pd.read_csv(file, sep=";")

        # ===============================
        # PREVIEW DATA
        # ===============================
        st.subheader("🔍 Preview Dataset")
        st.dataframe(data.head())
        st.write("Jumlah data:", data.shape[0])
        st.write("Jumlah fitur:", data.shape[1])

        # ===============================
        # VISUALISASI KORELASI
        # ===============================
        st.subheader("📊 Korelasi Antar Atribut")

        corr = data.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax_corr
        )
        st.pyplot(fig_corr)

        # ===============================
        # PAIRPLOT ATRIBUT UTAMA
        # ===============================
        st.subheader("🔍 Pairplot Antar Atribut Utama")

        selected_features = ["Glucose", "BMI", "Age", "Insulin"]
        pairplot_data = data[selected_features + [data.columns[-1]]]

        fig_pair = sns.pairplot(
            pairplot_data,
            hue=data.columns[-1],
            diag_kind="kde"
        )
        st.pyplot(fig_pair)

        # ===============================
        # PEMISAHAN FITUR & TARGET
        # ===============================
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # ===============================
        # PARAMETER MODEL
        # ===============================
        st.sidebar.subheader("⚙️ Parameter Model")
        kernel = st.sidebar.selectbox("Kernel SVM", ["linear", "rbf", "poly"])
        test_size = st.sidebar.slider("Porsi Data Uji", 0.1, 0.4, 0.2)

        # ===============================
        # PREPROCESSING
        # ===============================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ===============================
        # TRAINING MODEL
        # ===============================
        model = SVC(kernel=kernel)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # ===============================
        # EVALUASI MODEL
        # ===============================
        st.subheader("📈 Evaluasi Model")

        acc = accuracy_score(y_test, y_pred)
        st.metric("Akurasi Model", f"{acc:.2f}")

        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        st.pyplot(fig_cm)

        st.text(classification_report(y_test, y_pred))

        # ===============================
        # SIMPAN KE SESSION
        # ===============================
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["columns"] = X.columns

        st.success("✅ Model berhasil dilatih & siap digunakan!")

    else:
        st.info("Upload dataset terlebih dahulu.")

# ===============================
# PREDIKSI MANUAL
# ===============================
elif menu == "✍️ Prediksi Manual":
    st.title("✍️ Prediksi Diabetes (Input Manual)")

    if "model" not in st.session_state:
        st.warning("⚠️ Silakan training model terlebih dahulu.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 200, 120)
            blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
            skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

        with col2:
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 1, 100, 30)

        if st.button("🔍 Prediksi"):
            input_df = pd.DataFrame([[
                pregnancies, glucose, blood_pressure,
                skin_thickness, insulin, bmi, dpf, age
            ]], columns=st.session_state["columns"])

            scaler = st.session_state["scaler"]
            model = st.session_state["model"]

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            if prediction == 1:
                st.error("⚠️ HASIL: POSITIF DIABETES")
            else:
                st.success("✅ HASIL: NEGATIF DIABETES")
