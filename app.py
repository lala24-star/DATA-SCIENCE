import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Analisis Sentimen Pengunduran Diri Gibran", page_icon="📊", layout="wide")

SLANG_MAP = {
    "yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "tdk": "tidak", "bkn": "bukan", "blm": "belum", "udh": "sudah", "udah": "sudah",
    "ijasah": "ijazah", "persiden": "presiden", "bener": "benar", "krn": "karena",
    "tp": "tapi", "jd": "jadi", "jg": "juga", "aja": "saja", "ane": "saya",
}
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan",
    "adalah", "dalam", "oleh", "atau", "juga", "sudah", "akan", "ada", "saja",
    "saya", "kamu", "dia", "mereka", "kita", "kami", "nya", "ya", "nih", "sih",
    "dong", "kok", "kan", "kalau", "karena", "tapi", "jadi", "buat", "sama",
}


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    tokens = [SLANG_MAP.get(t, t) for t in text.split()]
    return " ".join(t for t in tokens if len(t) >= 2 and t not in STOPWORDS_ID)


@st.cache_resource
def load_model():
    return joblib.load("model_sentimen_opini_publik.joblib")


st.title("📊 Analisis Sentimen Opini Publik terhadap Pengunduran Diri Gibran")
st.caption("Model TF-IDF + Logistic Regression | Sumber Data: YouTube, Reddit, & Kaskus")

model = load_model()
tab1, tab2 = st.tabs(["Prediksi Komentar Tunggal", "Analisis Batch CSV"])

with tab1:
    teks = st.text_area("Masukkan komentar publik:", height=120, placeholder="Contoh: Setuju, keputusan tepat dan berani...")
    if st.button("Prediksi Sentimen", type="primary"):
        if teks.strip():
            clean = preprocess_text(teks)
            pred = model.predict([clean])[0]
            proba = dict(zip(model.classes_, model.predict_proba([clean])[0]))
            st.success(f"Hasil Klasifikasi: **{pred}** (Keyakinan: {proba[pred]*100:.1f}%)")
            st.bar_chart(pd.Series(proba, name="Probabilitas"))
        else:
            st.warning("Silakan masukkan teks komentar terlebih dahulu.")

with tab2:
    uploaded = st.file_uploader("Unggah file CSV (wajib memiliki kolom 'Komentar')", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        if "Komentar" not in df_up.columns:
            st.error("Kolom 'Komentar' tidak ditemukan di file CSV.")
        else:
            df_up["Komentar_Clean"] = df_up["Komentar"].fillna("").apply(preprocess_text)
            df_up["Prediksi_Sentimen"] = model.predict(df_up["Komentar_Clean"])
            st.subheader("Distribusi Sentimen")
            st.bar_chart(df_up["Prediksi_Sentimen"].value_counts())
            if "Platform" in df_up.columns:
                st.subheader("Perbandingan Sentimen per Platform (%)")
                ct = pd.crosstab(df_up["Platform"], df_up["Prediksi_Sentimen"], normalize="index") * 100
                st.dataframe(ct.round(2), use_container_width=True)
            st.download_button(
                "Unduh Hasil Prediksi (CSV)",
                df_up.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_sentimen.csv",
                mime="text/csv",
            )
