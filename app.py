from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Analisis Sentimen Opini Publik & Penyaringan Bot",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_FEATURE_COLS = [
    "jumlah_komentar",
    "jumlah_source_unik",
    "rentang_aktivitas_jam",
    "rata_interval_jam",
    "frekuensi_per_jam",
    "rasio_duplikasi",
    "rata_panjang_karakter",
    "keragaman_teks",
]

SAMPLE_COMMENTS = {
    "Contoh Kontra (YouTube)": (
        "YouTube",
        "Sebaiknya mundur saja supaya gaduh politik cepat selesai dan pemerintahan bisa fokus bekerja untuk rakyat.",
    ),
    "Contoh Pro (YouTube)": (
        "YouTube",
        "Tetap lanjut sampai akhir periode, kan dipilih secara konstitusional oleh rakyat lewat pemilu.",
    ),
    "Contoh Netral (Reddit)": (
        "Reddit",
        "Secara konstitusi mekanisme pemakzulan atau pengunduran diri wakil presiden sudah diatur jelas di UUD 1945.",
    ),
    "Contoh Bahasa Santai (Kaskus)": (
        "Kaskus",
        "Gak usah ribut terus gan, yg penting harga sembako turun dan lapangan kerja makin banyak.",
    ),
}


# ============================================================================
# HELPER PENCARIAN & PEMUATAN ARTEFAK (ROOT REPO ATAU FOLDER artifacts/)
# ============================================================================
def find_artifact(filename: str) -> Path | None:
    """Cari file di root repo terlebih dahulu, lalu di subfolder artifacts/."""
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "artifacts" / filename,
        Path(filename),
        Path("artifacts") / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_sentiment_artifact() -> tuple[Any | None, Path | None, str | None]:
    path = find_artifact("sentiment_model.joblib")
    if path is None:
        return None, None, "File `sentiment_model.joblib` tidak ditemukan di root repo maupun folder `artifacts/`."
    try:
        obj = joblib.load(path)
        return obj, path, None
    except Exception as exc:
        return None, path, f"Gagal memuat `sentiment_model.joblib`: {exc}"


@st.cache_resource(show_spinner=False)
def load_account_artifact() -> tuple[Any | None, Path | None, str | None]:
    path = find_artifact("account_model.joblib")
    if path is None:
        return None, None, "File `account_model.joblib` tidak ditemukan di root repo maupun folder `artifacts/`."
    try:
        obj = joblib.load(path)
        return obj, path, None
    except Exception as exc:
        return None, path, f"Gagal memuat `account_model.joblib`: {exc}"


@st.cache_data(show_spinner=False)
def load_metrics_json() -> tuple[dict[str, Any] | None, Path | None, str | None]:
    path = find_artifact("metrics.json")
    if path is None:
        return None, None, "File `metrics.json` tidak ditemukan di root repo maupun folder `artifacts/`."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data, path, None
    except Exception as exc:
        return None, path, f"Gagal membaca `metrics.json`: {exc}"


# ============================================================================
# PREPROCESSING & INFERENSI JALUR B (SENTIMEN)
# ============================================================================
def preprocess_indonesian_text(
    text: str,
    slang_map: dict[str, str] | None = None,
    stopwords: set[str] | list[str] | None = None,
    negation_words: set[str] | list[str] | None = None,
    normalize_slang: bool = True,
    remove_stopwords: bool = True,
) -> str:
    """Normalisasi teks komentar bahasa Indonesia sesuai pipeline notebook TAS."""
    s = str(text or "").lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"@[a-z0-9_.-]+", " ", s)
    s = re.sub(r"#([a-z0-9_]+)", r" \1 ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"(.)\1{2,}", r"\1", s)
    tokens = s.split()

    slang = slang_map or {}
    sw = set(stopwords or [])
    neg = set(negation_words or [])

    cleaned_tokens: list[str] = []
    for tok in tokens:
        if normalize_slang and tok in slang:
            tok = slang[tok]
        if remove_stopwords and tok in sw and tok not in neg:
            continue
        if len(tok) > 1 or tok in neg:
            cleaned_tokens.append(tok)
    return " ".join(cleaned_tokens)


def predict_sentiment(text: str, artifact: Any) -> dict[str, Any]:
    """Jalankan prediksi sentimen (mendukung format dict TAS maupun Pipeline sklearn)."""
    if isinstance(artifact, dict):
        vectorizer = artifact.get("vectorizer")
        classifier = artifact.get("classifier")
        slang_map = artifact.get("slang_map") or {}
        stopwords = artifact.get("stopwords") or set()
        negation_words = artifact.get("negation_words") or set()
        normalize_slang = bool(artifact.get("normalize_slang", True))
        remove_stopwords = bool(artifact.get("remove_stopwords", True))
        threshold = float(artifact.get("confidence_threshold", 0.60))
        model_name = str(artifact.get("model_name", type(classifier).__name__))

        cleaned = preprocess_indonesian_text(
            text,
            slang_map=slang_map,
            stopwords=stopwords,
            negation_words=negation_words,
            normalize_slang=normalize_slang,
            remove_stopwords=remove_stopwords,
        )
        X = vectorizer.transform([cleaned if cleaned else text.lower()])
        classes = [
            str(c)
            for c in (
                artifact.get("classes")
                if artifact.get("classes") is not None
                else getattr(classifier, "classes_", [])
            )
        ]

        probs_dict: dict[str, float] = {}
        if hasattr(classifier, "predict_proba"):
            probs = np.asarray(classifier.predict_proba(X)[0], dtype=float)
            best_idx = int(np.argmax(probs))
            pred_label = classes[best_idx] if classes else str(classifier.predict(X)[0])
            confidence = float(probs[best_idx])
            probs_dict = {
                classes[i]: float(probs[i]) for i in range(min(len(classes), len(probs)))
            }
        elif hasattr(classifier, "decision_function"):
            scores = np.asarray(classifier.decision_function(X)[0], dtype=float)
            if scores.ndim == 0:
                scores = np.array([-float(scores), float(scores)])
            shifted = scores - np.max(scores)
            exp_scores = np.exp(shifted)
            probs = exp_scores / np.sum(exp_scores)
            best_idx = int(np.argmax(probs))
            pred_label = classes[best_idx] if classes else str(classifier.predict(X)[0])
            confidence = float(probs[best_idx])
            probs_dict = {
                classes[i]: float(probs[i]) for i in range(min(len(classes), len(probs)))
            }
        else:
            pred_label = str(classifier.predict(X)[0])
            confidence = 1.0
            probs_dict = {pred_label: 1.0}

        return {
            "sentiment": pred_label,
            "confidence": confidence,
            "confidence_threshold": threshold,
            "needs_manual_review": confidence < threshold,
            "cleaned_text": cleaned,
            "probabilities": probs_dict,
            "model_name": model_name,
        }

    # Fallback jika artifact adalah sklearn Pipeline langsung
    cleaned = preprocess_indonesian_text(text, normalize_slang=False, remove_stopwords=False)
    pred_label = str(artifact.predict([cleaned])[0])
    probs_dict = {}
    confidence = 1.0
    if hasattr(artifact, "predict_proba"):
        probs = np.asarray(artifact.predict_proba([cleaned])[0], dtype=float)
        classes = [str(c) for c in getattr(artifact, "classes_", [])]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        probs_dict = {
            classes[i]: float(probs[i]) for i in range(min(len(classes), len(probs)))
        }
    return {
        "sentiment": pred_label,
        "confidence": confidence,
        "confidence_threshold": 0.60,
        "needs_manual_review": confidence < 0.60,
        "cleaned_text": cleaned,
        "probabilities": probs_dict,
        "model_name": type(artifact).__name__,
    }


# ============================================================================
# EKSTRAKSI FITUR & PENYARINGAN JALUR A (BOT / ANOMALI PERILAKU)
# ============================================================================
def extract_account_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Ekstrak 8 fitur perilaku per akun sesuai Jalur A di notebook TAS."""
    df = df_in.copy()

    # Normalisasi nama kolom case-insensitive agar fleksibel terhadap variasi CSV
    col_map = {c.strip().lower(): c for c in df.columns}
    required = {"platform": "Platform", "username": "Username", "komentar": "Komentar"}
    missing = [std for key, std in required.items() if key not in col_map]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di CSV: {', '.join(missing)}. "
            "Pastikan CSV memiliki kolom Platform, Username, dan Komentar."
        )

    rename_dict = {col_map[k]: v for k, v in required.items()}
    if "tanggal" in col_map:
        rename_dict[col_map["tanggal"]] = "Tanggal"
    if "source_id" in col_map:
        rename_dict[col_map["source_id"]] = "Source_ID"
    df = df.rename(columns=rename_dict)

    if "Tanggal" not in df.columns:
        df["Tanggal"] = pd.NaT
    if "Source_ID" not in df.columns:
        df["Source_ID"] = "default_source"

    # Bersihkan baris kosong & anomali header literal
    df = df.dropna(subset=["Komentar", "Username"]).copy()
    df["Username_Str"] = df["Username"].astype(str).str.strip()
    df = df[df["Username_Str"].str.lower() != "username"].copy()
    df["Komentar_Str"] = df["Komentar"].astype(str).str.strip()
    df = df[df["Komentar_Str"].str.len() > 0].copy()

    df["Platform"] = df["Platform"].astype(str).str.strip()
    df["Username_Norm"] = df["Username_Str"].str.lstrip("@").str.lower().str.strip()
    df["Account_ID"] = df["Platform"] + " | " + df["Username_Norm"]
    df["Komentar_Norm"] = df["Komentar_Str"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df["Tanggal_DT"] = pd.to_datetime(df["Tanggal"], errors="coerce")

    records: list[dict[str, Any]] = []
    for acc_id, g in df.groupby("Account_ID", sort=False):
        n_comments = int(len(g))
        n_sources = int(g["Source_ID"].astype(str).nunique())

        valid_ts = g["Tanggal_DT"].dropna().sort_values()
        if len(valid_ts) >= 2:
            span_hours = float((valid_ts.iloc[-1] - valid_ts.iloc[0]).total_seconds() / 3600.0)
            diffs_hours = valid_ts.diff().dropna().dt.total_seconds() / 3600.0
            mean_interval = float(diffs_hours.mean()) if len(diffs_hours) > 0 else 0.0
        else:
            span_hours = 0.0
            mean_interval = 0.0

        freq_per_hour = float(n_comments / max(span_hours, 1.0 / 60.0)) if n_comments > 1 else 0.0

        n_unique_comments = int(g["Komentar_Norm"].nunique())
        dup_ratio = float(1.0 - (n_unique_comments / n_comments)) if n_comments > 0 else 0.0
        avg_char_len = float(g["Komentar_Str"].str.len().mean())

        all_words = " ".join(g["Komentar_Norm"].tolist()).split()
        text_diversity = float(len(set(all_words)) / len(all_words)) if all_words else 0.0

        records.append(
            {
                "Account_ID": acc_id,
                "Platform": str(g["Platform"].iloc[0]),
                "jumlah_komentar": n_comments,
                "jumlah_source_unik": n_sources,
                "rentang_aktivitas_jam": round(span_hours, 4),
                "rata_interval_jam": round(mean_interval, 4),
                "frekuensi_per_jam": round(freq_per_hour, 4),
                "rasio_duplikasi": round(dup_ratio, 4),
                "rata_panjang_karakter": round(avg_char_len, 2),
                "keragaman_teks": round(text_diversity, 4),
            }
        )

    return pd.DataFrame(records)


def run_account_screening(df_raw: pd.DataFrame, account_artifact: dict[str, Any]) -> pd.DataFrame:
    """Terapkan IsolationForest + KMeans pada akun yang memenuhi syarat (>1 komentar)."""
    df_acc = extract_account_features(df_raw)
    if df_acc.empty:
        return df_acc

    df_acc["Status_Kecukupan"] = np.where(
        df_acc["jumlah_komentar"] == 1,
        "Data tidak cukup untuk dinilai",
        "Memenuhi syarat pemodelan (>1 komentar)",
    )
    df_acc["anomaly_score"] = np.nan
    df_acc["kmeans_cluster"] = pd.Series([pd.NA] * len(df_acc), dtype="Int64")
    df_acc["Status_Akun"] = "Data tidak cukup untuk dinilai"

    eligible_mask = df_acc["jumlah_komentar"] > 1
    if not eligible_mask.any():
        return df_acc

    feature_cols = list(account_artifact.get("feature_cols") or DEFAULT_FEATURE_COLS)
    for col in feature_cols:
        if col not in df_acc.columns:
            df_acc[col] = 0.0

    X = df_acc.loc[eligible_mask, feature_cols].fillna(0.0).to_numpy(dtype=float)

    skewed_idx = account_artifact.get("skewed_idx")
    if skewed_idx is None:
        skewed_cols = account_artifact.get("skewed_cols") or []
        skewed_idx = [feature_cols.index(c) for c in skewed_cols if c in feature_cols]

    X_trans = X.copy()
    for idx in skewed_idx:
        if 0 <= int(idx) < X_trans.shape[1]:
            X_trans[:, int(idx)] = np.log1p(np.clip(X_trans[:, int(idx)], 0.0, None))

    scaler = account_artifact.get("scaler")
    X_scaled = scaler.transform(X_trans) if scaler is not None else X_trans

    iso_model = account_artifact.get("iso_model")
    kmeans_model = account_artifact.get("kmeans_model")
    km_rep_cluster = account_artifact.get("km_repetitive_cluster")

    iso_preds = iso_model.predict(X_scaled) if iso_model is not None else np.ones(len(X_scaled), dtype=int)
    iso_scores = (
        iso_model.decision_function(X_scaled)
        if iso_model is not None and hasattr(iso_model, "decision_function")
        else np.zeros(len(X_scaled), dtype=float)
    )
    km_clusters = (
        kmeans_model.predict(X_scaled)
        if kmeans_model is not None
        else np.full(len(X_scaled), -1, dtype=int)
    )

    statuses: list[str] = []
    for iso_p, km_c in zip(iso_preds, km_clusters):
        is_iso_anom = int(iso_p) == -1
        is_rep = km_rep_cluster is not None and int(km_c) == int(km_rep_cluster)
        if is_iso_anom and is_rep:
            statuses.append("Indikasi Bot / Anomali Kuat")
        elif is_iso_anom:
            statuses.append("Indikasi Anomali Perilaku")
        elif is_rep:
            statuses.append("Pola Repetitif (K-Means)")
        else:
            statuses.append("Normal / Organik")

    df_acc.loc[eligible_mask, "anomaly_score"] = np.round(iso_scores, 4)
    df_acc.loc[eligible_mask, "kmeans_cluster"] = pd.Series(
        km_clusters, index=df_acc.index[eligible_mask], dtype="Int64"
    )
    df_acc.loc[eligible_mask, "Status_Akun"] = statuses

    return df_acc


def format_pct(val: Any, digits: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    try:
        num = float(val)
        return f"{(num * 100 if num <= 1 else num):.{digits}f}%"
    except Exception:
        return str(val)


# ============================================================================
# MUAT ARTEFAK & HEADER APLIKASI
# ============================================================================
sentiment_artifact, sent_path, sent_err = load_sentiment_artifact()
account_artifact, acc_path, acc_err = load_account_artifact()
metrics_data, metrics_path, metrics_err = load_metrics_json()

st.title("📊 Analisis Sentimen Opini Publik & Penyaringan Akun Bot")
st.caption(
    "Studi Kasus: Wacana Pengunduran Diri Gibran Rakabuming Raka di YouTube, Reddit, dan Kaskus "
    "(Metodologi CRISP-DM • Jalur A: IsolationForest + K-Means | Jalur B: Klasifikasi Sentimen)"
)

with st.sidebar:
    st.header("Status Artefak Model")
    st.write(f"• **Sentimen (Jalur B):** {'✅ Aktif' if sentiment_artifact is not None else '❌ Tidak ditemukan'}")
    st.write(f"• **Bot Screening (Jalur A):** {'✅ Aktif' if account_artifact is not None else '❌ Tidak ditemukan'}")
    st.write(f"• **Metrik Evaluasi:** {'✅ Aktif' if metrics_data is not None else '❌ Tidak ditemukan'}")
    if sent_path:
        st.caption(f"Model sentimen: `{sent_path.name}`")
    if acc_path:
        st.caption(f"Model akun: `{acc_path.name}`")
    if metrics_path:
        st.caption(f"Metrik: `{metrics_path.name}`")

tab_sentimen, tab_bot, tab_metrik = st.tabs(
    [
        "1. Prediksi Sentimen (Jalur B)",
        "2. Penyaringan Akun Bot CSV (Jalur A)",
        "3. Ringkasan Metrik & Visualisasi",
    ]
)

# ============================================================================
# TAB 1: PREDIKSI SENTIMEN
# ============================================================================
with tab_sentimen:
    st.subheader("Uji Klasifikasi Sentimen Komentar")
    if sent_err:
        st.error(sent_err)
    else:
        threshold_val = (
            float(sentiment_artifact.get("confidence_threshold", 0.60))
            if isinstance(sentiment_artifact, dict)
            else 0.60
        )
        model_label = (
            str(sentiment_artifact.get("model_name", "Model Sentimen"))
            if isinstance(sentiment_artifact, dict)
            else type(sentiment_artifact).__name__
        )
        st.info(f"Model aktif: **{model_label}** | Ambang batas confidence: **{threshold_val:.2f}**")

        # Tombol contoh komentar cepat
        if "input_komentar" not in st.session_state:
            st.session_state["input_komentar"] = ""
        if "input_platform" not in st.session_state:
            st.session_state["input_platform"] = "YouTube"

        sample_cols = st.columns(len(SAMPLE_COMMENTS))
        for col, (btn_label, (plat, txt)) in zip(sample_cols, SAMPLE_COMMENTS.items()):
            if col.button(btn_label, use_container_width=True):
                st.session_state["input_komentar"] = txt
                st.session_state["input_platform"] = plat

        c_plat, c_text = st.columns([1, 3])
        with c_plat:
            platform_opt = ["YouTube", "Reddit", "Kaskus"]
            default_idx = (
                platform_opt.index(st.session_state["input_platform"])
                if st.session_state["input_platform"] in platform_opt
                else 0
            )
            platform = st.selectbox("Platform Sumber", platform_opt, index=default_idx)
        with c_text:
            komentar = st.text_area(
                "Masukkan teks komentar publik",
                key="input_komentar",
                height=110,
                placeholder="Tulis atau tempel komentar dari YouTube, Reddit, atau Kaskus...",
            )

        if st.button("Prediksi Sentimen", type="primary"):
            if not komentar.strip():
                st.warning("Masukkan teks komentar terlebih dahulu.")
            else:
                res = predict_sentiment(komentar, sentiment_artifact)
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Prediksi Sentimen", res["sentiment"])
                col_res2.metric("Confidence", format_pct(res["confidence"], 1))
                col_res3.metric("Platform", platform)

                if res["needs_manual_review"]:
                    st.warning(
                        f"Confidence ({format_pct(res['confidence'], 1)}) berada di bawah threshold "
                        f"**{res['confidence_threshold']:.2f}**. Komentar ini ditandai **Perlu Tinjauan Manual**."
                    )
                else:
                    st.success("Confidence memenuhi ambang batas klasifikasi otomatis.")

                if res["probabilities"]:
                    prob_df = pd.DataFrame(
                        {
                            "Kelas": list(res["probabilities"].keys()),
                            "Probabilitas": list(res["probabilities"].values()),
                        }
                    ).sort_values("Probabilitas", ascending=False)
                    st.bar_chart(prob_df.set_index("Kelas"))

                with st.expander("Lihat hasil preprocessing teks"):
                    st.code(res["cleaned_text"] or "(kosong setelah pembersihan)", language="text")


# ============================================================================
# TAB 2: PENYARINGAN AKUN BOT CSV
# ============================================================================
with tab_bot:
    st.subheader("Penyaringan Akun Bot / Anomali Perilaku (Jalur A)")
    st.write(
        "Unggah file CSV komentar (`Platform`, `Username`, `Komentar`, `Tanggal`, `Source_ID`) "
        "untuk mengekstrak 8 fitur perilaku dan menjalankan model **IsolationForest + K-Means**. "
        "Akun dengan hanya 1 komentar otomatis diberi label *Data tidak cukup untuk dinilai*."
    )

    if acc_err:
        st.error(acc_err)
    else:
        uploaded_csv = st.file_uploader("Unggah file CSV komentar", type=["csv"])

        # Opsi uji langsung dengan dataset yang sudah ada di repo GitHub
        repo_csv = find_artifact("dataset_clean_master.csv") or find_artifact(
            "Data Scraping Opini Publik.csv"
        )
        use_repo_csv = False
        if repo_csv is not None:
            use_repo_csv = st.checkbox(
                f"Atau gunakan langsung dataset dari repo: `{repo_csv.name}`",
                value=False,
            )

        df_input: pd.DataFrame | None = None
        if uploaded_csv is not None:
            try:
                df_input = pd.read_csv(uploaded_csv)
            except Exception as exc:
                st.error(f"Gagal membaca file CSV yang diunggah: {exc}")
        elif use_repo_csv and repo_csv is not None:
            try:
                df_input = pd.read_csv(repo_csv)
            except Exception as exc:
                st.error(f"Gagal membaca `{repo_csv.name}`: {exc}")

        if df_input is not None:
            st.caption(f"Dataset terbaca: **{len(df_input):,}** baris × **{len(df_input.columns)}** kolom.")
            if st.button("Jalankan Penyaringan Akun Bot", type="primary"):
                try:
                    with st.spinner("Mengekstrak fitur perilaku & menjalankan model Jalur A..."):
                        df_screened = run_account_screening(df_input, account_artifact)
                    st.session_state["df_screened"] = df_screened
                except Exception as exc:
                    st.error(f"Proses penyaringan gagal: {exc}")

        if "df_screened" in st.session_state:
            df_res: pd.DataFrame = st.session_state["df_screened"]
            total_acc = len(df_res)
            eligible_acc = int((df_res["jumlah_komentar"] > 1).sum())
            insuff_acc = int((df_res["jumlah_komentar"] == 1).sum())
            flagged_acc = int(
                df_res["Status_Akun"].str.contains("Bot|Anomali|Repetitif", case=False, na=False).sum()
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Akun Unik", f"{total_acc:,}")
            m2.metric("Memenuhi Syarat (>1 Komentar)", f"{eligible_acc:,}")
            m3.metric("Terindikasi Bot / Anomali", f"{flagged_acc:,}")
            m4.metric("Data Tidak Cukup (1 Komentar)", f"{insuff_acc:,}")

            f1, f2, f3 = st.columns([2, 1, 1])
            q_search = f1.text_input("Cari Account_ID / Username", "")
            plat_choices = ["Semua"] + sorted(df_res["Platform"].dropna().unique().tolist())
            plat_sel = f2.selectbox("Filter Platform", plat_choices)
            status_choices = ["Semua"] + sorted(df_res["Status_Akun"].dropna().unique().tolist())
            status_sel = f3.selectbox("Filter Status Akun", status_choices)

            df_view = df_res.copy()
            if q_search.strip():
                df_view = df_view[
                    df_view["Account_ID"].str.contains(q_search.strip(), case=False, na=False)
                ]
            if plat_sel != "Semua":
                df_view = df_view[df_view["Platform"] == plat_sel]
            if status_sel != "Semua":
                df_view = df_view[df_view["Status_Akun"] == status_sel]

            st.dataframe(df_view.head(500), use_container_width=True, hide_index=True)
            if len(df_view) > 500:
                st.caption(
                    f"Menampilkan 500 baris pertama dari {len(df_view):,} akun terfilter. "
                    "Unduh CSV untuk melihat seluruh hasil."
                )

            csv_bytes = df_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Unduh Hasil Penyaringan ({len(df_view):,} akun) sebagai CSV",
                data=csv_bytes,
                file_name="hasil_penyaringan_akun_bot.csv",
                mime="text/csv",
            )


# ============================================================================
# TAB 3: RINGKASAN METRIK & VISUALISASI
# ============================================================================
with tab_metrik:
    st.subheader("Ringkasan Evaluasi Model (metrics.json)")
    if metrics_err:
        st.warning(metrics_err)
    elif metrics_data is not None:
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### Jalur A — Penyaringan Akun Bot")
            jalur_a = metrics_data.get("jalur_a_selected") or {}
            if isinstance(jalur_a, dict) and jalur_a:
                st.write(f"**Model terpilih:** {jalur_a.get('model_name', 'IsolationForest + K-Means (k=3)')}")
                a_rows = [
                    {"Metrik": k, "Nilai": v}
                    for k, v in jalur_a.items()
                    if not isinstance(v, (dict, list)) and k != "model_name"
                ]
                if a_rows:
                    st.dataframe(pd.DataFrame(a_rows), use_container_width=True, hide_index=True)
            else:
                st.write(f"**Model terpilih:** {jalur_a or 'IsolationForest + K-Means (k=3)'}")

        with c_right:
            st.markdown("#### Jalur B — Klasifikasi Sentimen")
            jalur_b = metrics_data.get("jalur_b_selected") or {}
            if isinstance(jalur_b, dict) and jalur_b:
                st.write(f"**Model terpilih:** {jalur_b.get('model_name', '-')}")
                b_rows = [
                    {"Metrik": k, "Nilai": v}
                    for k, v in jalur_b.items()
                    if not isinstance(v, (dict, list)) and k != "model_name"
                ]
                if b_rows:
                    st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)
            else:
                st.write(f"**Model terpilih:** {jalur_b or '-'}")

            # Tampilkan perbandingan 80/20 vs 70/30 jika tersedia
            split_80 = (
                (jalur_b.get("split_80_20") if isinstance(jalur_b, dict) else None)
                or metrics_data.get("split_80_20")
            )
            split_70 = (
                (jalur_b.get("split_70_30") if isinstance(jalur_b, dict) else None)
                or metrics_data.get("split_70_30")
            )
            if isinstance(split_80, dict) or isinstance(split_70, dict):
                s80 = split_80 or {}
                s70 = split_70 or {}
                keys = sorted(set(s80.keys()) | set(s70.keys()))
                comp_df = pd.DataFrame(
                    [
                        {
                            "Metrik": k,
                            "Split 80/20": format_pct(s80.get(k)) if isinstance(s80.get(k), (int, float)) else str(s80.get(k, "-")),
                            "Split 70/30": format_pct(s70.get(k)) if isinstance(s70.get(k), (int, float)) else str(s70.get(k, "-")),
                        }
                        for k in keys
                    ]
                )
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        with st.expander("Lihat isi lengkap metrics.json"):
            st.json(metrics_data)

    # Galeri grafik EDA & hasil evaluasi yang ada di repo GitHub
    chart_files = [
        ("Distribusi Platform", "eda_1_distribusi_platform.png"),
        ("Tren Komentar November 2026", "eda_2_tren_november_2026.png"),
        ("Aktivitas Akun", "eda_3_aktivitas_akun.png"),
        ("Panjang Komentar", "eda_4_panjang_komentar.png"),
        ("Benang Merah Jalur A × Jalur B", "benang_merah_jalurA_x_jalurB.png"),
    ]
    found_charts = [(title, find_artifact(fname)) for title, fname in chart_files]
    found_charts = [(title, p) for title, p in found_charts if p is not None]

    if found_charts:
        st.markdown("#### Visualisasi EDA & Analisis di Repositori")
        cols = st.columns(2)
        for idx, (title, img_path) in enumerate(found_charts):
            with cols[idx % 2]:
                st.image(str(img_path), caption=title, use_container_width=True)
