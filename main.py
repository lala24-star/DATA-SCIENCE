"""
Backend FastAPI Siap Deploy ke Render (https://render.com)
Proyek: Analisis Sentimen Opini Publik — Gibran Rakabuming Raka
AM552 Data Science of IS, FTI UKSW

Menghubungkan artefak notebook TAS_682024037_fix.ipynb ke Frontend Next.js (Vercel):
  - sentiment_model.joblib (Jalur B)
  - account_model.joblib   (Jalur A)
  - metrics.json           (Ringkasan Evaluasi)
  - dataset_clean_master.csv / Data Scraping Opini Publik.csv (opsional untuk Demo 1-Klik)

Start Command di Render:
  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

# ============================================================================
# INISIALISASI FASTAPI & CORS (IZINKAN VERCEL & LOCALHOST)
# ============================================================================
app = FastAPI(
    title="API Analisis Sentimen Opini Publik & Penyaringan Bot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPER PENCARIAN ARTEFAK (ROOT REPO ATAU SUBFOLDER artifacts/)
# ============================================================================
def find_artifact(filename: str) -> Path | None:
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


class ArtifactStore:
    def __init__(self) -> None:
        self.sentiment_artifact: Any | None = None
        self.account_artifact: Any | None = None
        self.metrics: dict[str, Any] = {}
        self._integration_cache: dict[str, Any] | None = None
        self._screen_demo_cache: dict[str, Any] | None = None
        self.load_all()

    def load_all(self) -> None:
        sent_path = find_artifact("sentiment_model.joblib")
        if sent_path is not None:
            self.sentiment_artifact = joblib.load(sent_path)

        acc_path = find_artifact("account_model.joblib")
        if acc_path is not None:
            self.account_artifact = joblib.load(acc_path)

        metrics_path = find_artifact("metrics.json")
        if metrics_path is not None:
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))


STORE = ArtifactStore()


# ============================================================================
# PREPROCESSING BAHASA INDONESIA (PERSIS CELL 15 NOTEBOOK TAS)
# ============================================================================
def normalize_elongated(word: str) -> str:
    if word in {"aamiin", "amiin"}:
        return "aamiin"
    return re.sub(r"(.)\1{2,}", r"\1", word)


def preprocess_indonesian_text(
    text: str,
    slang_map: dict[str, str] | None = None,
    stopwords: set[str] | list[str] | None = None,
    remove_stopwords: bool = True,
    normalize_slang: bool = True,
) -> str:
    t = str(text or "").lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@[A-Za-z0-9_.-]+", " ", t)
    t = re.sub(r"!+", " tandaseru ", t)
    t = re.sub(r"\?+", " tandatanya ", t)
    t = re.sub(r"\b([a-z]+)2(an)?\b", r"\1", t)
    t = re.sub(r"[^a-z\s]", " ", t)

    raw_tokens = [normalize_elongated(w) for w in t.split()]

    slang = slang_map or {}
    sw = set(stopwords or [])

    if normalize_slang:
        expanded: list[str] = []
        for w in raw_tokens:
            mapped = slang.get(w, w)
            expanded.extend(str(mapped).split())
        tokens = expanded
    else:
        tokens = raw_tokens

    if remove_stopwords:
        tokens = [w for w in tokens if w not in sw and len(w) > 1]
    else:
        tokens = [w for w in tokens if len(w) > 1]

    return " ".join(tokens)


def extract_detected_keywords(cleaned_text: str, vectorizer: Any, top_k: int = 5) -> list[str]:
    """Ambil kata/frasa TF-IDF terkuat yang terdeteksi pada komentar."""
    if not cleaned_text or vectorizer is None:
        return []
    try:
        vec = vectorizer.transform([cleaned_text])
        if vec.nnz == 0:
            return []
        feature_names = vectorizer.get_feature_names_out()
        indices = vec.indices
        data = vec.data
        order = np.argsort(data)[::-1][:top_k]
        return [str(feature_names[indices[i]]) for i in order]
    except Exception:
        return []


def predict_single_sentiment(
    text: str,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    artifact = STORE.sentiment_artifact
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Artefak sentiment_model.joblib tidak ditemukan di server.",
        )

    if isinstance(artifact, dict):
        vectorizer = artifact["vectorizer"]
        classifier = artifact["classifier"]
        slang_map = artifact.get("slang_map") or {}
        stopwords = artifact.get("stopwords") or set()
        normalize_slang = bool(artifact.get("normalize_slang", True))
        remove_stopwords = bool(artifact.get("remove_stopwords", True))
        default_threshold = float(artifact.get("confidence_threshold", 0.60))
        threshold = float(threshold_override) if threshold_override is not None else default_threshold
        model_name = str(artifact.get("model_name", type(classifier).__name__))

        cleaned = preprocess_indonesian_text(
            text,
            slang_map=slang_map,
            stopwords=stopwords,
            remove_stopwords=remove_stopwords,
            normalize_slang=normalize_slang,
        )
        X = vectorizer.transform([cleaned if cleaned else str(text).lower()])
        classes = [
            str(c)
            for c in (
                artifact.get("classes")
                if artifact.get("classes") is not None
                else getattr(classifier, "classes_", [])
            )
        ]

        if hasattr(classifier, "predict_proba"):
            probs = np.asarray(classifier.predict_proba(X)[0], dtype=float)
            best_idx = int(np.argmax(probs))
            pred_label = classes[best_idx] if classes else str(classifier.predict(X)[0])
            confidence = float(probs[best_idx])
            probs_dict = {
                classes[i]: round(float(probs[i]), 4)
                for i in range(min(len(classes), len(probs)))
            }
        else:
            pred_label = str(classifier.predict(X)[0])
            confidence = 1.0
            probs_dict = {pred_label: 1.0}

        needs_review = confidence < threshold
        status_ops = (
            f"Perlu Tinjauan Manual (Confidence < {threshold:.2f})"
            if needs_review
            else "Keyakinan Memadai"
        )
        keywords = extract_detected_keywords(cleaned, vectorizer, top_k=5)

        return {
            "sentiment": pred_label,
            "confidence": round(confidence, 4),
            "confidence_threshold": round(threshold, 2),
            "needs_manual_review": needs_review,
            "status_operasional": status_ops,
            "cleaned_text": cleaned,
            "detected_keywords": keywords,
            "probabilities": probs_dict,
            "model_name": model_name,
        }

    raise HTTPException(
        status_code=500,
        detail="Format sentiment_model.joblib tidak sesuai dengan dictionary artefak TAS.",
    )


# ============================================================================
# EKSTRAKSI FITUR & PENYARINGAN JALUR A (PERSIS NOTEBOOK TAS)
# ============================================================================
def extract_account_features(df_input: pd.DataFrame) -> pd.DataFrame:
    data = df_input.copy()
    col_map = {c.strip().lower(): c for c in data.columns}
    required = {"platform": "Platform", "username": "Username", "komentar": "Komentar"}
    missing = [std for key, std in required.items() if key not in col_map]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di CSV: {', '.join(missing)}. "
            "Kolom minimal: Platform, Username, Komentar."
        )

    rename_dict = {col_map[k]: v for k, v in required.items()}
    if "tanggal" in col_map:
        rename_dict[col_map["tanggal"]] = "Tanggal"
    if "source_id" in col_map:
        rename_dict[col_map["source_id"]] = "Source_ID"
    data = data.rename(columns=rename_dict)

    if "Tanggal" not in data.columns:
        data["Tanggal"] = pd.NaT
    if "Source_ID" not in data.columns:
        data["Source_ID"] = "default_source"

    data = data.dropna(subset=["Komentar", "Username"]).copy()
    data["Username_Norm"] = data["Username"].astype(str).str.strip()
    data = data[data["Username_Norm"].str.lower() != "username"].copy()
    data["Komentar_Str"] = data["Komentar"].astype(str).str.strip()
    data = data[data["Komentar_Str"].str.len() > 0].copy()

    data["Platform"] = data["Platform"].astype(str).str.strip()
    m_kaskus = data["Platform"] == "Kaskus"
    data.loc[m_kaskus, "Username_Norm"] = data.loc[m_kaskus, "Username_Norm"].str.lstrip("@")

    data["Account_ID"] = data["Platform"] + " | " + data["Username_Norm"]
    data["Komentar_Norm"] = (
        data["Komentar_Str"].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    )
    data["Char_Len"] = data["Komentar_Str"].str.len()
    data["Tanggal_dt"] = pd.to_datetime(data["Tanggal"], errors="coerce")

    data_sorted = data.sort_values(["Account_ID", "Tanggal_dt"])
    records: list[dict[str, Any]] = []

    for acc_id, g in data_sorted.groupby("Account_ID", sort=False):
        n_comments = int(len(g))
        n_sources = int(g["Source_ID"].astype(str).nunique())

        times = g["Tanggal_dt"].dropna().values
        if len(times) > 1:
            span_hours = float((times[-1] - times[0]) / np.timedelta64(1, "h"))
            diffs_sec = np.diff(times) / np.timedelta64(1, "s")
            mean_interval_hours = float(np.mean(diffs_sec)) / 3600.0
        else:
            span_hours = 0.0
            mean_interval_hours = 0.0

        freq_per_hour = float(n_comments / max(span_hours, 1.0))
        n_unique_texts = int(g["Komentar_Norm"].nunique())
        dup_ratio = float(1.0 - (n_unique_texts / n_comments)) if n_comments > 0 else 0.0
        all_tokens = " ".join(g["Komentar_Norm"].tolist()).split()
        text_diversity = float(len(set(all_tokens)) / len(all_tokens)) if all_tokens else 0.0
        mean_char_len = float(g["Char_Len"].mean())

        records.append(
            {
                "Account_ID": acc_id,
                "Platform": str(g["Platform"].iloc[0]),
                "Username": str(g["Username_Norm"].iloc[0]),
                "jumlah_komentar": n_comments,
                "jumlah_source_unik": n_sources,
                "rentang_aktivitas_jam": round(span_hours, 4),
                "rata_interval_jam": round(mean_interval_hours, 4),
                "frekuensi_per_jam": round(freq_per_hour, 4),
                "rasio_duplikasi": round(dup_ratio, 4),
                "rata_panjang_karakter": round(mean_char_len, 2),
                "keragaman_teks": round(text_diversity, 4),
            }
        )

    return pd.DataFrame(records)


def score_accounts_dataframe(df_raw: pd.DataFrame) -> dict[str, Any]:
    acc_art = STORE.account_artifact
    if acc_art is None:
        raise HTTPException(
            status_code=503,
            detail="Artefak account_model.joblib tidak ditemukan di server.",
        )

    df_acc = extract_account_features(df_raw)
    if df_acc.empty:
        return {
            "summary": {
                "total_accounts": 0,
                "insufficient_data_accounts": 0,
                "normal_accounts": 0,
                "flagged_bot_accounts": 0,
            },
            "accounts": [],
        }

    df_acc["Status_Kecukupan"] = np.where(
        df_acc["jumlah_komentar"] == 1,
        "Data Tidak Cukup (1 Komentar)",
        "Memenuhi Syarat (>1 Komentar)",
    )
    df_acc["skor_if"] = None
    df_acc["kmeans_cluster"] = None
    df_acc["Status_Akun"] = "Data Tidak Cukup (1 Komentar)"

    eligible_mask = df_acc["jumlah_komentar"] > 1
    if eligible_mask.any():
        feature_cols = list(acc_art.get("feature_cols") or DEFAULT_FEATURE_COLS)
        skewed_idx = acc_art.get("skewed_idx") or []
        X = df_acc.loc[eligible_mask, feature_cols].fillna(0.0).to_numpy(dtype=float).copy()
        for idx in skewed_idx:
            if 0 <= int(idx) < X.shape[1]:
                X[:, int(idx)] = np.log1p(np.maximum(X[:, int(idx)], 0.0))

        X_scaled = acc_art["scaler"].transform(X)
        iso_preds = acc_art["iso_model"].predict(X_scaled)
        iso_scores = -acc_art["iso_model"].decision_function(X_scaled)
        km_clusters = acc_art["kmeans_model"].predict(X_scaled)

        statuses = np.where(
            iso_preds == -1,
            "Terindikasi Bot / Anomali",
            "Aktif Normal (>1 Komentar)",
        )
        df_acc.loc[eligible_mask, "skor_if"] = np.round(iso_scores, 4)
        df_acc.loc[eligible_mask, "kmeans_cluster"] = [int(c) for c in km_clusters]
        df_acc.loc[eligible_mask, "Status_Akun"] = statuses

    # Urutkan prioritas audit: Terindikasi Bot/Anomali di atas berdasarkan Skor IF tertinggi
    df_acc["_priority"] = np.where(
        df_acc["Status_Akun"] == "Terindikasi Bot / Anomali",
        0,
        np.where(df_acc["Status_Akun"] == "Aktif Normal (>1 Komentar)", 1, 2),
    )
    df_acc["_sort_score"] = pd.to_numeric(df_acc["skor_if"], errors="coerce").fillna(-999.0)
    df_acc = df_acc.sort_values(["_priority", "_sort_score"], ascending=[True, False]).drop(
        columns=["_priority", "_sort_score"]
    )

    total_acc = int(len(df_acc))
    insuff_acc = int((df_acc["jumlah_komentar"] == 1).sum())
    flagged_acc = int((df_acc["Status_Akun"] == "Terindikasi Bot / Anomali").sum())
    normal_acc = int((df_acc["Status_Akun"] == "Aktif Normal (>1 Komentar)").sum())

    # Batasi payload JSON agar tetap ringan di browser (kirim semua akun anomali + sampel normal/single hingga 1.500 baris)
    df_flagged = df_acc[df_acc["Status_Akun"] == "Terindikasi Bot / Anomali"]
    df_rest = df_acc[df_acc["Status_Akun"] != "Terindikasi Bot / Anomali"].head(
        max(0, 1500 - len(df_flagged))
    )
    df_payload = pd.concat([df_flagged, df_rest], ignore_index=True)

    accounts_list = df_payload.where(pd.notna(df_payload), None).to_dict(orient="records")

    return {
        "summary": {
            "total_accounts": total_acc,
            "insufficient_data_accounts": insuff_acc,
            "normal_accounts": normal_acc,
            "flagged_bot_accounts": flagged_acc,
        },
        "accounts": accounts_list,
    }


def load_project_dataset() -> pd.DataFrame:
    path = find_artifact("dataset_clean_master.csv") or find_artifact(
        "Data Scraping Opini Publik.csv"
    )
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset proyek (dataset_clean_master.csv / Data Scraping Opini Publik.csv) tidak ditemukan di server.",
        )
    return pd.read_csv(path)


# ============================================================================
# SKEMA REQUEST PYDANTIC
# ============================================================================
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)
    platform: str | None = "YouTube"
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SimulateAccountRequest(BaseModel):
    platform: str = "YouTube"
    username: str = "@akun_uji_coba"
    n_sources: int = Field(default=1, ge=1, le=500)
    comments: list[str] = Field(..., min_length=1)


# ============================================================================
# ENDPOINTS API
# ============================================================================
@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "sentiment_loaded": STORE.sentiment_artifact is not None,
        "account_loaded": STORE.account_artifact is not None,
        "metrics_loaded": bool(STORE.metrics),
    }


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    payload = dict(STORE.metrics)
    if STORE.sentiment_artifact is not None and isinstance(STORE.sentiment_artifact, dict):
        payload.setdefault("jalur_b_selected", STORE.sentiment_artifact.get("model_name"))
        payload.setdefault(
            "confidence_threshold",
            STORE.sentiment_artifact.get("confidence_threshold", 0.60),
        )
        if "test_metrics" in STORE.sentiment_artifact:
            tm = STORE.sentiment_artifact["test_metrics"]
            payload.setdefault("jalur_b_test_macro_f1_80_20", tm.get("macro_f1"))
            payload.setdefault("jalur_b_test_accuracy_80_20", tm.get("accuracy"))
            payload.setdefault("jalur_b_test_recall_negatif_80_20", tm.get("recall_negatif"))
    if STORE.account_artifact is not None and isinstance(STORE.account_artifact, dict):
        payload.setdefault("jalur_a_selected", STORE.account_artifact.get("model_name"))
    return payload


@app.post("/predict-sentiment")
def api_predict_sentiment(req: SentimentRequest) -> dict[str, Any]:
    return predict_single_sentiment(req.text, threshold_override=req.threshold)


@app.post("/predict-batch")
async def api_predict_batch(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.45, ge=0.0, le=1.0),
) -> dict[str, Any]:
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV tidak dapat dibaca: {exc}")

    col_map = {c.strip().lower(): c for c in df.columns}
    if "komentar" not in col_map:
        raise HTTPException(status_code=400, detail="CSV wajib memiliki kolom 'Komentar'.")

    kom_col = col_map["komentar"]
    plat_col = col_map.get("platform")
    df_valid = df.dropna(subset=[kom_col]).head(1000).copy()

    rows: list[dict[str, Any]] = []
    counts = {"Positif": 0, "Negatif": 0, "Netral": 0}
    for _, row in df_valid.iterrows():
        text = str(row[kom_col]).strip()
        if not text:
            continue
        res = predict_single_sentiment(text, threshold_override=threshold)
        sent = res["sentiment"]
        counts[sent] = counts.get(sent, 0) + 1
        rows.append(
            {
                "Platform": str(row[plat_col]) if plat_col else "-",
                "Komentar": text[:240],
                "Sentimen": sent,
                "Confidence": res["confidence"],
                "Status": res["status_operasional"],
            }
        )

    return {
        "total_processed": len(rows),
        "distribution": counts,
        "rows": rows,
    }


@app.get("/predict-demo")
def api_predict_demo(
    threshold: float = Query(default=0.45, ge=0.0, le=1.0),
    limit: int = Query(default=493, ge=10, le=1000),
) -> dict[str, Any]:
    df = load_project_dataset()
    df_valid = df.dropna(subset=["Komentar"]).copy()
    if len(df_valid) > limit:
        df_sample = df_valid.sample(n=limit, random_state=42)
    else:
        df_sample = df_valid

    rows: list[dict[str, Any]] = []
    counts = {"Positif": 0, "Negatif": 0, "Netral": 0}
    for _, row in df_sample.iterrows():
        text = str(row["Komentar"]).strip()
        if not text:
            continue
        res = predict_single_sentiment(text, threshold_override=threshold)
        sent = res["sentiment"]
        counts[sent] = counts.get(sent, 0) + 1
        rows.append(
            {
                "Platform": str(row.get("Platform", "-")),
                "Komentar": text[:240],
                "Sentimen": sent,
                "Confidence": res["confidence"],
                "Status": res["status_operasional"],
            }
        )

    return {
        "total_processed": len(rows),
        "distribution": counts,
        "rows": rows,
    }


@app.post("/screen-accounts")
async def api_screen_accounts(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
        return score_accounts_dataframe(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memproses penyaringan: {exc}")


@app.get("/screen-demo")
def api_screen_demo() -> dict[str, Any]:
    if STORE._screen_demo_cache is not None:
        return STORE._screen_demo_cache
    df = load_project_dataset()
    result = score_accounts_dataframe(df)
    STORE._screen_demo_cache = result
    return result


@app.post("/simulate-account")
def api_simulate_account(req: SimulateAccountRequest) -> dict[str, Any]:
    comments = [c.strip() for c in req.comments if c.strip()]
    if not comments:
        raise HTTPException(status_code=400, detail="Masukkan minimal 1 komentar.")

    n_comments = len(comments)
    now = pd.Timestamp.now()
    rows = []
    for i, text in enumerate(comments):
        src_idx = (i % max(1, req.n_sources)) + 1
        rows.append(
            {
                "Platform": req.platform,
                "Source_ID": f"SIM_SRC_{src_idx}",
                "Tanggal": (now + pd.Timedelta(seconds=i * 20)).strftime("%Y-%m-%d %H:%M:%S"),
                "Username": req.username,
                "Komentar": text,
            }
        )

    df_sim = pd.DataFrame(rows)
    res = score_accounts_dataframe(df_sim)
    account_row = res["accounts"][0] if res["accounts"] else None
    return {
        "account": account_row,
        "note": (
            "Akun dengan 1 komentar otomatis berstatus Data Tidak Cukup."
            if n_comments == 1
            else "Evaluasi selesai menggunakan model IsolationForest + K-Means (Jalur A)."
        ),
    }


@app.get("/integration-summary")
def api_integration_summary() -> dict[str, Any]:
    """Jalankan Integrasi Benang Merah Jalur A x Jalur B (Cell 17 Notebook TAS)."""
    if STORE._integration_cache is not None:
        return STORE._integration_cache

    if STORE.sentiment_artifact is None or STORE.account_artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model Jalur A dan Jalur B wajib tersedia untuk analisis integrasi.",
        )

    df_master = load_project_dataset().dropna(subset=["Komentar", "Username"]).copy()
    sent_art = STORE.sentiment_artifact
    acc_art = STORE.account_artifact

    # 1. Prediksi sentimen seluruh korpus
    slang_map = sent_art.get("slang_map") or {}
    stopwords = sent_art.get("stopwords") or set()
    remove_sw = bool(sent_art.get("remove_stopwords", True))
    norm_slang = bool(sent_art.get("normalize_slang", True))

    texts_clean = df_master["Komentar"].apply(
        lambda s: preprocess_indonesian_text(
            s,
            slang_map=slang_map,
            stopwords=stopwords,
            remove_stopwords=remove_sw,
            normalize_slang=norm_slang,
        )
    )
    X_vec = sent_art["vectorizer"].transform(texts_clean)
    df_master["Pred_Sentimen"] = sent_art["classifier"].predict(X_vec)

    # 2. Petakan status akun Jalur A
    df_master["Username_Norm"] = df_master["Username"].astype(str).str.strip()
    m_k = df_master["Platform"] == "Kaskus"
    df_master.loc[m_k, "Username_Norm"] = df_master.loc[m_k, "Username_Norm"].str.lstrip("@")
    df_master["Account_ID"] = df_master["Platform"].astype(str) + " | " + df_master["Username_Norm"]

    df_acc = extract_account_features(df_master)
    eligible = df_acc[df_acc["jumlah_komentar"] > 1].copy()

    feature_cols = list(acc_art.get("feature_cols") or DEFAULT_FEATURE_COLS)
    skewed_idx = acc_art.get("skewed_idx") or []
    X_f = eligible[feature_cols].fillna(0.0).to_numpy(dtype=float).copy()
    for idx in skewed_idx:
        if 0 <= int(idx) < X_f.shape[1]:
            X_f[:, int(idx)] = np.log1p(np.maximum(X_f[:, int(idx)], 0.0))
    X_f_scaled = acc_art["scaler"].transform(X_f)
    iso_flags = acc_art["iso_model"].predict(X_f_scaled)

    status_map = dict(
        zip(
            eligible["Account_ID"],
            np.where(iso_flags == -1, "Terindikasi Bot/Anomali", "Aktif Normal (>1 Komentar)"),
        )
    )
    df_master["Status_Akun_JalurA"] = (
        df_master["Account_ID"].map(status_map).fillna("Data Tidak Cukup (1 Komentar)")
    )

    # 3. Sebelum vs Sesudah Filter Bot
    classes = ["Positif", "Negatif", "Netral"]
    before_pct = (df_master["Pred_Sentimen"].value_counts(normalize=True) * 100).to_dict()
    df_after = df_master[df_master["Status_Akun_JalurA"] != "Terindikasi Bot/Anomali"]
    after_pct = (df_after["Pred_Sentimen"].value_counts(normalize=True) * 100).to_dict()

    comparison = []
    for cls in classes:
        b = round(float(before_pct.get(cls, 0.0)), 2)
        a = round(float(after_pct.get(cls, 0.0)), 2)
        comparison.append(
            {
                "sentimen": cls,
                "sebelum_pct": b,
                "sesudah_pct": a,
                "pergeseran_poin": round(a - b, 2),
            }
        )

    # Proporsi sentimen per kategori akun
    ct_pct = pd.crosstab(
        df_master["Status_Akun_JalurA"], df_master["Pred_Sentimen"], normalize="index"
    ) * 100
    ct_cnt = df_master["Status_Akun_JalurA"].value_counts().to_dict()

    by_category = []
    for cat in [
        "Data Tidak Cukup (1 Komentar)",
        "Aktif Normal (>1 Komentar)",
        "Terindikasi Bot/Anomali",
    ]:
        if cat in ct_pct.index:
            by_category.append(
                {
                    "kategori": cat,
                    "jumlah_komentar": int(ct_cnt.get(cat, 0)),
                    "positif_pct": round(float(ct_pct.loc[cat].get("Positif", 0.0)), 2),
                    "negatif_pct": round(float(ct_pct.loc[cat].get("Negatif", 0.0)), 2),
                    "netral_pct": round(float(ct_pct.loc[cat].get("Netral", 0.0)), 2),
                }
            )

    result = {
        "total_comments_before": int(len(df_master)),
        "total_comments_after": int(len(df_after)),
        "filtered_out_comments": int(len(df_master) - len(df_after)),
        "comparison": comparison,
        "by_category": by_category,
    }
    STORE._integration_cache = result
    return result