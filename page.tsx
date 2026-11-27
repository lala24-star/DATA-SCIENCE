"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";

/**
 * Frontend Next.js (App Router) — Siap Deploy ke Vercel
 * Proyek: Analisis Sentimen Opini Publik & Penyaringan Akun Bot (YouTube, Reddit, Kaskus)
 *
 * Cara pasang di project Next.js (Tailwind CSS aktif):
 *   1. Simpan file ini sebagai `src/app/page.tsx` (atau `app/page.tsx`).
 *   2. Tambahkan environment variable di `.env.local` dan di Vercel Dashboard:
 *        NEXT_PUBLIC_API_URL=https://<nama-backend-fastapi-anda>.onrender.com
 *
 * Kontrak Endpoint FastAPI yang dihubungkan:
 *   - POST /predict-sentiment
 *       Request JSON : { "text": string, "platform"?: string }
 *       Response JSON: {
 *         "sentiment": string,
 *         "confidence": number,
 *         "confidence_threshold"?: number,
 *         "needs_manual_review"?: boolean,
 *         "cleaned_text"?: string,
 *         "probabilities"?: Record<string, number>,
 *         "model_name"?: string
 *       }
 *
 *   - POST /screen-accounts
 *       Request Form : multipart/form-data dengan field "file" (.csv)
 *       Response JSON: {
 *         "summary": {
 *           "total_accounts": number,
 *           "eligible_accounts": number,
 *           "insufficient_data_accounts": number,
 *           "flagged_bot_accounts": number,
 *           "normal_accounts"?: number
 *         },
 *         "accounts": Array<{
 *           "Account_ID": string,
 *           "Platform": string,
 *           "jumlah_komentar": number,
 *           "jumlah_source_unik"?: number,
 *           "frekuensi_per_jam"?: number,
 *           "rasio_duplikasi": number,
 *           "keragaman_teks"?: number,
 *           "rata_panjang_karakter"?: number,
 *           "anomaly_score"?: number | null,
 *           "kmeans_cluster"?: number | null,
 *           "Status_Kecukupan": string,
 *           "Status_Akun": string
 *         }>
 *       }
 *
 *   - GET /metrics
 *       Response JSON: isi file `artifacts/metrics.json` dari notebook TAS
 */

// ============================================================================
// KONFIGURASI API
// ============================================================================
const RAW_API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_BASE_URL = RAW_API_URL.replace(/\/+$/, "");

// ============================================================================
// TIPE DATA TYPESCRIPT
// ============================================================================
type ActiveTab = "sentiment" | "bot" | "metrics";

interface SentimentResponse {
  sentiment: string;
  confidence: number;
  confidence_threshold?: number;
  needs_manual_review?: boolean;
  cleaned_text?: string;
  probabilities?: Record<string, number>;
  model_name?: string;
}

interface SentimentHistoryItem {
  id: string;
  text: string;
  platform: string;
  sentiment: string;
  confidence: number;
  needsReview: boolean;
  timestamp: string;
}

interface ScreenedAccount {
  Account_ID: string;
  Platform: string;
  jumlah_komentar: number;
  jumlah_source_unik?: number;
  rentang_aktivitas_jam?: number;
  rata_interval_jam?: number;
  frekuensi_per_jam?: number;
  rasio_duplikasi: number;
  rata_panjang_karakter?: number;
  keragaman_teks?: number;
  anomaly_score?: number | null;
  kmeans_cluster?: number | null;
  Status_Kecukupan: string;
  Status_Akun: string;
}

interface ScreenAccountsSummary {
  total_accounts: number;
  eligible_accounts: number;
  insufficient_data_accounts: number;
  flagged_bot_accounts: number;
  normal_accounts?: number;
}

interface ScreenAccountsResponse {
  summary: ScreenAccountsSummary;
  accounts: ScreenedAccount[];
}

interface SplitMetric {
  macro_f1?: number;
  accuracy?: number;
  recall_negatif?: number;
  recall_negative?: number;
  f1_macro?: number;
}

interface MetricsPayload {
  jalur_a_selected?: {
    model_name?: string;
    silhouette_score?: number;
    davies_bouldin_index?: number;
    ari_iso_stability?: number;
    ari_kmeans_stability?: number;
    contamination?: number;
    n_clusters?: number;
    total_accounts?: number;
    eligible_accounts?: number;
    insufficient_accounts?: number;
  };
  jalur_b_selected?: {
    model_name?: string;
    target_macro_f1?: number;
    confidence_threshold?: number;
    split_80_20?: SplitMetric;
    split_70_30?: SplitMetric;
  };
  confidence_threshold?: number;
  split_80_20?: SplitMetric;
  split_70_30?: SplitMetric;
  dataset?: {
    raw_rows?: number;
    clean_rows?: number;
    unique_accounts?: number;
    platforms?: Record<string, number>;
  };
  [key: string]: unknown;
}

// ============================================================================
// DATA CONTOH KOMENTAR UNTUK UJI CEPAT
// ============================================================================
const SAMPLE_COMMENTS: Array<{
  label: string;
  platform: string;
  text: string;
}> = [
  {
    label: "Contoh Kontra",
    platform: "YouTube",
    text: "Sebaiknya mundur saja supaya gaduh politik cepat selesai dan pemerintahan bisa fokus bekerja untuk rakyat.",
  },
  {
    label: "Contoh Pro",
    platform: "YouTube",
    text: "Tetap lanjut sampai akhir periode, kan dipilih secara konstitusional oleh rakyat lewat pemilu.",
  },
  {
    label: "Contoh Netral",
    platform: "Reddit",
    text: "Secara konstitusi mekanisme pemakzulan atau pengunduran diri wakil presiden sudah diatur jelas di UUD 1945.",
  },
  {
    label: "Contoh Bahasa Santai",
    platform: "Kaskus",
    text: "Gak usah ribut terus gan, yg penting harga sembako turun dan lapangan kerja makin banyak.",
  },
];

// ============================================================================
// HELPER FORMATTING & BADGE WARNA
// ============================================================================
function formatPercent(value: number | undefined | null, digits = 1): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "-";
  }
  const normalized = value <= 1 ? value * 100 : value;
  return `${normalized.toFixed(digits)}%`;
}

function formatNumber(value: number | undefined | null, digits = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "-";
  }
  return value.toLocaleString("id-ID", {
    maximumFractionDigits: digits,
  });
}

function getSentimentBadgeStyle(label: string): {
  bg: string;
  text: string;
  border: string;
  bar: string;
} {
  const norm = label.trim().toLowerCase();
  if (
    norm.includes("pro") ||
    norm.includes("positif") ||
    norm.includes("mendukung")
  ) {
    return {
      bg: "bg-emerald-50",
      text: "text-emerald-800",
      border: "border-emerald-200",
      bar: "bg-emerald-600",
    };
  }
  if (
    norm.includes("kontra") ||
    norm.includes("negatif") ||
    norm.includes("menolak")
  ) {
    return {
      bg: "bg-rose-50",
      text: "text-rose-800",
      border: "border-rose-200",
      bar: "bg-rose-600",
    };
  }
  return {
    bg: "bg-amber-50",
    text: "text-amber-800",
    border: "border-amber-200",
    bar: "bg-amber-500",
  };
}

function getAccountStatusBadge(status: string): string {
  const norm = status.toLowerCase();
  if (norm.includes("bot") || norm.includes("anomali") || norm.includes("curiga")) {
    return "bg-rose-50 text-rose-800 border-rose-200";
  }
  if (norm.includes("repetitif") || norm.includes("waspada")) {
    return "bg-amber-50 text-amber-800 border-amber-200";
  }
  if (norm.includes("tidak cukup")) {
    return "bg-slate-100 text-slate-600 border-slate-200";
  }
  return "bg-emerald-50 text-emerald-800 border-emerald-200";
}

function downloadSampleCsv(): void {
  const csvRows = [
    "Platform,Source_ID,Tanggal,Username,Komentar,Jumlah_Like",
    'YouTube,YT001,2026-11-10 08:01:00,@akun_spam01,"Mundur sekarang juga!!",0',
    'YouTube,YT001,2026-11-10 08:01:25,@akun_spam01,"Mundur sekarang juga!!",0',
    'YouTube,YT002,2026-11-10 08:02:10,@akun_spam01,"Mundur sekarang juga!!",0',
    'YouTube,YT003,2026-11-10 08:03:00,@akun_spam01,"Mundur sekarang juga!!",0',
    'Reddit,RD001,2026-11-11 10:15:00,warga_diskusi,"Menurut saya perlu dilihat dari sisi konstitusi dan mekanisme DPR.",12',
    'Reddit,RD002,2026-11-12 19:40:00,warga_diskusi,"Setuju, debat kebijakan jauh lebih penting daripada sekadar ribut personal.",8',
    'Kaskus,KS001,2026-11-13 14:22:00,kaskuser_satu,"Menyimak dulu perkembangannya gan.",2',
  ];
  const blob = new Blob([csvRows.join("\n")], {
    type: "text/csv;charset=utf-8;",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "contoh_format_screening_bot.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function exportScreenedCsv(accounts: ScreenedAccount[]): void {
  const headers = [
    "Account_ID",
    "Platform",
    "jumlah_komentar",
    "jumlah_source_unik",
    "frekuensi_per_jam",
    "rasio_duplikasi",
    "keragaman_teks",
    "anomaly_score",
    "kmeans_cluster",
    "Status_Kecukupan",
    "Status_Akun",
  ];
  const escapeCell = (val: unknown): string => {
    const str = val === null || val === undefined ? "" : String(val);
    return `"${str.replace(/"/g, '""')}"`;
  };
  const lines = [
    headers.join(","),
    ...accounts.map((row) =>
      headers
        .map((key) => escapeCell(row[key as keyof ScreenedAccount]))
        .join(",")
    ),
  ];
  const blob = new Blob([lines.join("\n")], {
    type: "text/csv;charset=utf-8;",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "hasil_penyaringan_akun_bot.csv";
  link.click();
  URL.revokeObjectURL(url);
}

// ============================================================================
// KOMPONEN UTAMA
// ============================================================================
export default function OpiniPublikDashboardPage() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("sentiment");
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">(
    "checking"
  );

  // --- State Menu 1: Prediksi Sentimen ---
  const [commentText, setCommentText] = useState<string>("");
  const [selectedPlatform, setSelectedPlatform] = useState<string>("YouTube");
  const [sentimentLoading, setSentimentLoading] = useState<boolean>(false);
  const [sentimentError, setSentimentError] = useState<string | null>(null);
  const [sentimentResult, setSentimentResult] =
    useState<SentimentResponse | null>(null);
  const [sentimentHistory, setSentimentHistory] = useState<
    SentimentHistoryItem[]
  >([]);

  // --- State Menu 2: Penyaringan Akun Bot CSV ---
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [botLoading, setBotLoading] = useState<boolean>(false);
  const [botError, setBotError] = useState<string | null>(null);
  const [botResult, setBotResult] = useState<ScreenAccountsResponse | null>(
    null
  );
  const [searchAccount, setSearchAccount] = useState<string>("");
  const [filterPlatform, setFilterPlatform] = useState<string>("Semua");
  const [filterStatus, setFilterStatus] = useState<string>("Semua");

  // --- State Menu 3: Ringkasan Metrik ---
  const [metricsLoading, setMetricsLoading] = useState<boolean>(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [metricsData, setMetricsData] = useState<MetricsPayload | null>(null);

  // ==========================================================================
  // FETCH METRIK & CEK KONEKSI BACKEND
  // ==========================================================================
  const fetchMetrics = useCallback(async () => {
    setMetricsLoading(true);
    setMetricsError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/metrics`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(
          `Backend merespons status ${response.status} saat memuat /metrics.`
        );
      }
      const data: MetricsPayload = await response.json();
      setMetricsData(data);
      setApiStatus("online");
    } catch (err) {
      setApiStatus("offline");
      setMetricsError(
        err instanceof Error
          ? err.message
          : "Gagal terhubung ke backend FastAPI."
      );
    } finally {
      setMetricsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchMetrics();
  }, [fetchMetrics]);

  // ==========================================================================
  // HANDLER MENU 1: PREDIKSI SENTIMEN
  // ==========================================================================
  const handlePredictSentiment = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const trimmed = commentText.trim();
    if (!trimmed) {
      setSentimentError("Masukkan teks komentar terlebih dahulu.");
      return;
    }

    setSentimentLoading(true);
    setSentimentError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict-sentiment`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          text: trimmed,
          platform: selectedPlatform,
        }),
      });

      if (!response.ok) {
        const errBody = await response.text();
        throw new Error(
          `Prediksi gagal (${response.status}): ${errBody || response.statusText}`
        );
      }

      const data: SentimentResponse = await response.json();
      setSentimentResult(data);
      setApiStatus("online");

      const threshold = data.confidence_threshold ?? 0.6;
      const needsReview =
        data.needs_manual_review ?? data.confidence < threshold;

      setSentimentHistory((prev) => [
        {
          id: `${Date.now()}`,
          text: trimmed,
          platform: selectedPlatform,
          sentiment: data.sentiment,
          confidence: data.confidence,
          needsReview,
          timestamp: new Date().toLocaleTimeString("id-ID", {
            hour: "2-digit",
            minute: "2-digit",
          }),
        },
        ...prev.slice(0, 7),
      ]);
    } catch (err) {
      setSentimentError(
        err instanceof Error
          ? err.message
          : "Terjadi kesalahan saat memanggil endpoint /predict-sentiment."
      );
    } finally {
      setSentimentLoading(false);
    }
  };

  // ==========================================================================
  // HANDLER MENU 2: PENYARINGAN AKUN BOT CSV
  // ==========================================================================
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setCsvFile(file);
    setBotError(null);
  };

  const handleScreenAccounts = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!csvFile) {
      setBotError("Pilih file CSV komentar terlebih dahulu.");
      return;
    }

    setBotLoading(true);
    setBotError(null);

    try {
      const formData = new FormData();
      formData.append("file", csvFile);

      const response = await fetch(`${API_BASE_URL}/screen-accounts`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errBody = await response.text();
        throw new Error(
          `Penyaringan gagal (${response.status}): ${errBody || response.statusText}`
        );
      }

      const data: ScreenAccountsResponse = await response.json();
      setBotResult(data);
      setApiStatus("online");
    } catch (err) {
      setBotError(
        err instanceof Error
          ? err.message
          : "Terjadi kesalahan saat memanggil endpoint /screen-accounts."
      );
    } finally {
      setBotLoading(false);
    }
  };

  // Filter tabel hasil penyaringan akun bot
  const filteredAccounts = useMemo(() => {
    if (!botResult?.accounts) return [];
    const q = searchAccount.trim().toLowerCase();

    return botResult.accounts.filter((acc) => {
      const matchQuery =
        !q ||
        acc.Account_ID.toLowerCase().includes(q) ||
        acc.Platform.toLowerCase().includes(q) ||
        acc.Status_Akun.toLowerCase().includes(q);

      const matchPlatform =
        filterPlatform === "Semua" ||
        acc.Platform.toLowerCase().includes(filterPlatform.toLowerCase());

      const statusNorm = acc.Status_Akun.toLowerCase();
      const matchStatus =
        filterStatus === "Semua" ||
        (filterStatus === "Anomali/Bot" &&
          (statusNorm.includes("bot") ||
            statusNorm.includes("anomali") ||
            statusNorm.includes("repetitif"))) ||
        (filterStatus === "Normal" &&
          (statusNorm.includes("normal") || statusNorm.includes("organik"))) ||
        (filterStatus === "Tidak Cukup" &&
          (statusNorm.includes("tidak cukup") ||
            acc.Status_Kecukupan.toLowerCase().includes("tidak cukup")));

      return matchQuery && matchPlatform && matchStatus;
    });
  }, [botResult, searchAccount, filterPlatform, filterStatus]);

  // Ekstraksi metrik Jalur B (mendukung struktur nested maupun flat di metrics.json)
  const split80 =
    metricsData?.jalur_b_selected?.split_80_20 ?? metricsData?.split_80_20;
  const split70 =
    metricsData?.jalur_b_selected?.split_70_30 ?? metricsData?.split_70_30;
  const activeThreshold =
    metricsData?.jalur_b_selected?.confidence_threshold ??
    metricsData?.confidence_threshold ??
    0.6;
  const targetMacroF1 = metricsData?.jalur_b_selected?.target_macro_f1 ?? 0.75;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* ==================================================================== */}
      {/* HEADER                                                               */}
      {/* ==================================================================== */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600">
              Prototipe CRISP-DM • YouTube, Reddit &amp; Kaskus
            </p>
            <h1 className="mt-1 text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
              Analisis Sentimen Opini Publik &amp; Penyaringan Akun Bot
            </h1>
            <p className="mt-1 text-sm text-slate-600">
              Studi kasus wacana pengunduran diri Gibran Rakabuming Raka (Jalur A:
              IsolationForest + K-Means, Jalur B: Klasifikasi Sentimen).
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${
                apiStatus === "online"
                  ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                  : apiStatus === "offline"
                    ? "border-rose-200 bg-rose-50 text-rose-800"
                    : "border-slate-200 bg-slate-100 text-slate-700"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  apiStatus === "online"
                    ? "bg-emerald-500"
                    : apiStatus === "offline"
                      ? "bg-rose-500"
                      : "bg-slate-400"
                }`}
              />
              {apiStatus === "online"
                ? "FastAPI Terhubung"
                : apiStatus === "offline"
                  ? "FastAPI Belum Terhubung"
                  : "Memeriksa Backend..."}
            </span>
            <span
              className="max-w-[260px] truncate rounded-full border border-slate-200 bg-slate-50 px-3 py-1 font-mono text-xs text-slate-600"
              title={API_BASE_URL}
            >
              {API_BASE_URL}
            </span>
          </div>
        </div>

        {/* NAVIGASI 3 MENU */}
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <nav
            className="-mb-px flex gap-2 overflow-x-auto"
            aria-label="Menu Utama"
          >
            {[
              {
                id: "sentiment" as ActiveTab,
                label: "1. Prediksi Sentimen",
                sub: "Jalur B",
              },
              {
                id: "bot" as ActiveTab,
                label: "2. Penyaringan Akun Bot (CSV)",
                sub: "Jalur A",
              },
              {
                id: "metrics" as ActiveTab,
                label: "3. Ringkasan Metrik",
                sub: "Evaluasi Model",
              },
            ].map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex shrink-0 items-center gap-2 border-b-2 px-4 py-3 text-sm font-semibold transition-colors ${
                    isActive
                      ? "border-indigo-600 text-indigo-700"
                      : "border-transparent text-slate-600 hover:border-slate-300 hover:text-slate-900"
                  }`}
                >
                  <span>{tab.label}</span>
                  <span
                    className={`rounded-md px-1.5 py-0.5 text-[11px] font-medium ${
                      isActive
                        ? "bg-indigo-50 text-indigo-700"
                        : "bg-slate-100 text-slate-500"
                    }`}
                  >
                    {tab.sub}
                  </span>
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* ==================================================================== */}
      {/* KONTEN UTAMA                                                         */}
      {/* ==================================================================== */}
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        {/* ================================================================== */}
        {/* MENU 1: PREDIKSI SENTIMEN                                          */}
        {/* ================================================================== */}
        {activeTab === "sentiment" && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {/* Form Input */}
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm lg:col-span-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-lg font-bold text-slate-900">
                    Uji Prediksi Sentimen Komentar
                  </h2>
                  <p className="text-sm text-slate-600">
                    Komentar diproses melalui normalisasi slang &amp; negasi bahasa
                    Indonesia sebelum diklasifikasikan.
                  </p>
                </div>
                <span className="rounded-lg bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
                  Threshold: {activeThreshold.toFixed(2)}
                </span>
              </div>

              <form onSubmit={handlePredictSentiment} className="mt-5 space-y-4">
                <div>
                  <label
                    htmlFor="platform-select"
                    className="block text-sm font-medium text-slate-700"
                  >
                    Platform Sumber
                  </label>
                  <select
                    id="platform-select"
                    value={selectedPlatform}
                    onChange={(e) => setSelectedPlatform(e.target.value)}
                    className="mt-1.5 w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100 sm:w-56"
                  >
                    <option value="YouTube">YouTube</option>
                    <option value="Reddit">Reddit</option>
                    <option value="Kaskus">Kaskus</option>
                  </select>
                </div>

                <div>
                  <label
                    htmlFor="comment-input"
                    className="block text-sm font-medium text-slate-700"
                  >
                    Teks Komentar Publik
                  </label>
                  <textarea
                    id="comment-input"
                    rows={4}
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    placeholder="Tulis atau tempel komentar dari YouTube, Reddit, atau Kaskus di sini..."
                    className="mt-1.5 w-full rounded-xl border border-slate-300 bg-white p-3.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                  />
                </div>

                {/* Tombol contoh cepat */}
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-medium text-slate-500">
                    Coba contoh:
                  </span>
                  {SAMPLE_COMMENTS.map((sample) => (
                    <button
                      key={sample.label}
                      type="button"
                      onClick={() => {
                        setCommentText(sample.text);
                        setSelectedPlatform(sample.platform);
                        setSentimentError(null);
                      }}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-700 transition hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700"
                    >
                      {sample.label}
                    </button>
                  ))}
                </div>

                {sentimentError && (
                  <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
                    {sentimentError}
                  </div>
                )}

                <div className="flex items-center gap-3 pt-1">
                  <button
                    type="submit"
                    disabled={sentimentLoading}
                    className="inline-flex items-center justify-center rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {sentimentLoading
                      ? "Memproses Prediksi..."
                      : "Prediksi Sentimen"}
                  </button>
                  {commentText && (
                    <button
                      type="button"
                      onClick={() => {
                        setCommentText("");
                        setSentimentResult(null);
                        setSentimentError(null);
                      }}
                      className="rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-medium text-slate-600 hover:bg-slate-50"
                    >
                      Bersihkan
                    </button>
                  )}
                </div>
              </form>

              {/* Hasil Prediksi */}
              {sentimentResult && (
                <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                        Hasil Klasifikasi
                      </p>
                      <div className="mt-1.5 flex items-center gap-2.5">
                        <span
                          className={`inline-flex items-center rounded-full border px-3.5 py-1 text-sm font-bold ${
                            getSentimentBadgeStyle(sentimentResult.sentiment).bg
                          } ${
                            getSentimentBadgeStyle(sentimentResult.sentiment)
                              .text
                          } ${
                            getSentimentBadgeStyle(sentimentResult.sentiment)
                              .border
                          }`}
                        >
                          {sentimentResult.sentiment}
                        </span>
                        <span className="text-sm font-semibold text-slate-700">
                          Confidence: {formatPercent(sentimentResult.confidence)}
                        </span>
                      </div>
                    </div>

                    {sentimentResult.model_name && (
                      <span className="rounded-lg border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
                        Model: {sentimentResult.model_name}
                      </span>
                    )}
                  </div>

                  {/* Peringatan Confidence Threshold */}
                  {(sentimentResult.needs_manual_review ??
                    sentimentResult.confidence <
                      (sentimentResult.confidence_threshold ??
                        activeThreshold)) && (
                    <div className="mt-4 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-xs font-medium text-amber-900">
                      Confidence di bawah ambang batas{" "}
                      {(
                        sentimentResult.confidence_threshold ?? activeThreshold
                      ).toFixed(2)}
                      . Sesuai desain Jalur B, komentar ini ditandai{" "}
                      <strong>Perlu Tinjauan Manual</strong>.
                    </div>
                  )}

                  {/* Distribusi Probabilitas */}
                  {sentimentResult.probabilities &&
                    Object.keys(sentimentResult.probabilities).length > 0 && (
                      <div className="mt-5 space-y-2.5">
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                          Distribusi Probabilitas Kelas
                        </p>
                        {Object.entries(sentimentResult.probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([cls, prob]) => {
                            const style = getSentimentBadgeStyle(cls);
                            const pct = Math.max(
                              0,
                              Math.min(100, prob <= 1 ? prob * 100 : prob)
                            );
                            return (
                              <div key={cls}>
                                <div className="mb-1 flex justify-between text-xs font-medium text-slate-700">
                                  <span>{cls}</span>
                                  <span>{pct.toFixed(1)}%</span>
                                </div>
                                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                                  <div
                                    className={`h-full rounded-full transition-all ${style.bar}`}
                                    style={{ width: `${pct}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                      </div>
                    )}

                  {/* Teks Hasil Preprocessing */}
                  {sentimentResult.cleaned_text && (
                    <div className="mt-4 rounded-xl border border-slate-200 bg-white p-3">
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                        Hasil Preprocessing Teks
                      </p>
                      <p className="mt-1 font-mono text-xs text-slate-700">
                        {sentimentResult.cleaned_text}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </section>

            {/* Panel Samping: Riwayat Pengujian */}
            <aside className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-base font-bold text-slate-900">
                Riwayat Uji Sesi Ini
              </h3>
              <p className="mt-1 text-xs text-slate-500">
                Menyimpan hingga 8 prediksi terakhir selama halaman terbuka.
              </p>

              {sentimentHistory.length === 0 ? (
                <div className="mt-6 rounded-xl border border-dashed border-slate-200 p-6 text-center text-xs text-slate-500">
                  Belum ada komentar yang diuji.
                </div>
              ) : (
                <ul className="mt-4 space-y-3">
                  {sentimentHistory.map((item) => {
                    const style = getSentimentBadgeStyle(item.sentiment);
                    return (
                      <li
                        key={item.id}
                        className="rounded-xl border border-slate-200 bg-slate-50 p-3"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span
                            className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${style.bg} ${style.text} ${style.border}`}
                          >
                            {item.sentiment} • {formatPercent(item.confidence, 0)}
                          </span>
                          <span className="text-[11px] text-slate-400">
                            {item.platform} · {item.timestamp}
                          </span>
                        </div>
                        <p className="mt-2 line-clamp-2 text-xs text-slate-700">
                          {item.text}
                        </p>
                        {item.needsReview && (
                          <p className="mt-1 text-[11px] font-medium text-amber-700">
                            ⚠ Tinjauan manual disarankan
                          </p>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </aside>
          </div>
        )}

        {/* ================================================================== */}
        {/* MENU 2: PENYARINGAN AKUN BOT CSV                                   */}
        {/* ================================================================== */}
        {activeTab === "bot" && (
          <div className="space-y-6">
            {/* Upload Card */}
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <h2 className="text-lg font-bold text-slate-900">
                    Penyaringan Akun Bot / Anomali Perilaku (CSV)
                  </h2>
                  <p className="mt-1 text-sm text-slate-600">
                    Unggah file CSV komentar untuk mengekstrak 8 fitur perilaku
                    akun (Jalur A: IsolationForest + K-Means). Akun dengan 1
                    komentar otomatis ditandai{" "}
                    <span className="font-medium text-slate-800">
                      &ldquo;Data tidak cukup untuk dinilai&rdquo;
                    </span>
                    .
                  </p>
                </div>
                <button
                  type="button"
                  onClick={downloadSampleCsv}
                  className="rounded-xl border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50"
                >
                  Unduh Contoh CSV
                </button>
              </div>

              <form onSubmit={handleScreenAccounts} className="mt-5 space-y-4">
                <div className="rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 p-6 text-center">
                  <input
                    id="csv-upload"
                    type="file"
                    accept=".csv,text/csv"
                    onChange={handleFileChange}
                    className="mx-auto block w-full max-w-md text-sm text-slate-600 file:mr-4 file:rounded-lg file:border-0 file:bg-indigo-600 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-indigo-700"
                  />
                  <p className="mt-3 text-xs text-slate-500">
                    Kolom yang dibutuhkan:{" "}
                    <code className="rounded bg-slate-200 px-1.5 py-0.5 text-slate-800">
                      Platform
                    </code>
                    ,{" "}
                    <code className="rounded bg-slate-200 px-1.5 py-0.5 text-slate-800">
                      Username
                    </code>
                    ,{" "}
                    <code className="rounded bg-slate-200 px-1.5 py-0.5 text-slate-800">
                      Komentar
                    </code>
                    ,{" "}
                    <code className="rounded bg-slate-200 px-1.5 py-0.5 text-slate-800">
                      Tanggal
                    </code>
                    ,{" "}
                    <code className="rounded bg-slate-200 px-1.5 py-0.5 text-slate-800">
                      Source_ID
                    </code>
                  </p>
                  {csvFile && (
                    <p className="mt-2 text-xs font-semibold text-indigo-700">
                      File siap dianalisis: {csvFile.name} (
                      {(csvFile.size / 1024).toFixed(1)} KB)
                    </p>
                  )}
                </div>

                {botError && (
                  <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
                    {botError}
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="submit"
                    disabled={botLoading || !csvFile}
                    className="rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {botLoading
                      ? "Menjalankan Penyaringan Akun..."
                      : "Jalankan Penyaringan Bot"}
                  </button>
                  {botResult && (
                    <button
                      type="button"
                      onClick={() => exportScreenedCsv(filteredAccounts)}
                      className="rounded-xl border border-emerald-300 bg-emerald-50 px-4 py-2.5 text-sm font-semibold text-emerald-800 hover:bg-emerald-100"
                    >
                      Ekspor Hasil ({filteredAccounts.length}) ke CSV
                    </button>
                  )}
                </div>
              </form>
            </section>

            {/* Ringkasan & Tabel Hasil */}
            {botResult && (
              <>
                <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                  <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                    <p className="text-xs font-medium text-slate-500">
                      Total Akun Unik
                    </p>
                    <p className="mt-2 text-2xl font-bold text-slate-900">
                      {botResult.summary.total_accounts.toLocaleString("id-ID")}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                    <p className="text-xs font-medium text-slate-500">
                      Memenuhi Syarat (&gt;1 Komentar)
                    </p>
                    <p className="mt-2 text-2xl font-bold text-indigo-700">
                      {botResult.summary.eligible_accounts.toLocaleString(
                        "id-ID"
                      )}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-rose-200 bg-rose-50 p-5 shadow-sm">
                    <p className="text-xs font-medium text-rose-700">
                      Terindikasi Bot / Anomali
                    </p>
                    <p className="mt-2 text-2xl font-bold text-rose-800">
                      {botResult.summary.flagged_bot_accounts.toLocaleString(
                        "id-ID"
                      )}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                    <p className="text-xs font-medium text-slate-500">
                      Data Tidak Cukup (1 Komentar)
                    </p>
                    <p className="mt-2 text-2xl font-bold text-slate-700">
                      {botResult.summary.insufficient_data_accounts.toLocaleString(
                        "id-ID"
                      )}
                    </p>
                  </div>
                </div>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <h3 className="text-base font-bold text-slate-900">
                      Daftar Akun ({filteredAccounts.length.toLocaleString("id-ID")}{" "}
                      baris)
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      <input
                        type="text"
                        value={searchAccount}
                        onChange={(e) => setSearchAccount(e.target.value)}
                        placeholder="Cari Account_ID..."
                        className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs text-slate-900 focus:border-indigo-500 focus:outline-none"
                      />
                      <select
                        value={filterPlatform}
                        onChange={(e) => setFilterPlatform(e.target.value)}
                        className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-800"
                      >
                        <option value="Semua">Semua Platform</option>
                        <option value="YouTube">YouTube</option>
                        <option value="Reddit">Reddit</option>
                        <option value="Kaskus">Kaskus</option>
                      </select>
                      <select
                        value={filterStatus}
                        onChange={(e) => setFilterStatus(e.target.value)}
                        className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-800"
                      >
                        <option value="Semua">Semua Status</option>
                        <option value="Anomali/Bot">Anomali / Bot</option>
                        <option value="Normal">Normal</option>
                        <option value="Tidak Cukup">Data Tidak Cukup</option>
                      </select>
                    </div>
                  </div>

                  <div className="mt-4 overflow-x-auto">
                    <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                      <thead className="bg-slate-50 text-slate-600">
                        <tr>
                          <th className="px-3 py-3 font-semibold">Account_ID</th>
                          <th className="px-3 py-3 font-semibold">Platform</th>
                          <th className="px-3 py-3 text-right font-semibold">
                            Komentar
                          </th>
                          <th className="px-3 py-3 text-right font-semibold">
                            Rasio Duplikasi
                          </th>
                          <th className="px-3 py-3 text-right font-semibold">
                            Keragaman Teks
                          </th>
                          <th className="px-3 py-3 text-right font-semibold">
                            Frek / Jam
                          </th>
                          <th className="px-3 py-3 text-right font-semibold">
                            Cluster
                          </th>
                          <th className="px-3 py-3 font-semibold">
                            Status Akun
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white">
                        {filteredAccounts.slice(0, 150).map((acc) => (
                          <tr
                            key={acc.Account_ID}
                            className="hover:bg-slate-50"
                          >
                            <td className="max-w-[220px] truncate px-3 py-2.5 font-mono text-slate-900">
                              {acc.Account_ID}
                            </td>
                            <td className="px-3 py-2.5 text-slate-700">
                              {acc.Platform}
                            </td>
                            <td className="px-3 py-2.5 text-right font-medium text-slate-900">
                              {acc.jumlah_komentar}
                            </td>
                            <td className="px-3 py-2.5 text-right text-slate-700">
                              {formatPercent(acc.rasio_duplikasi)}
                            </td>
                            <td className="px-3 py-2.5 text-right text-slate-700">
                              {formatPercent(acc.keragaman_teks)}
                            </td>
                            <td className="px-3 py-2.5 text-right text-slate-700">
                              {formatNumber(acc.frekuensi_per_jam, 2)}
                            </td>
                            <td className="px-3 py-2.5 text-right text-slate-700">
                              {acc.kmeans_cluster ?? "-"}
                            </td>
                            <td className="px-3 py-2.5">
                              <span
                                className={`inline-flex rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${getAccountStatusBadge(
                                  acc.Status_Akun
                                )}`}
                              >
                                {acc.Status_Akun}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {filteredAccounts.length > 150 && (
                      <p className="mt-3 text-xs text-slate-500">
                        Menampilkan 150 baris pertama dari{" "}
                        {filteredAccounts.length.toLocaleString("id-ID")} akun.
                        Gunakan tombol &ldquo;Ekspor Hasil ke CSV&rdquo; untuk
                        mengunduh seluruh data.
                      </p>
                    )}
                  </div>
                </section>
              </>
            )}
          </div>
        )}

        {/* ================================================================== */}
        {/* MENU 3: RINGKASAN METRIK                                           */}
        {/* ================================================================== */}
        {activeTab === "metrics" && (
          <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-bold text-slate-900">
                  Ringkasan Evaluasi Model (artifacts/metrics.json)
                </h2>
                <p className="text-sm text-slate-600">
                  Metrik performa Jalur A (Unsupervised Bot Screening) dan Jalur
                  B (Supervised Sentiment Classification).
                </p>
              </div>
              <button
                type="button"
                onClick={() => void fetchMetrics()}
                disabled={metricsLoading}
                className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
              >
                {metricsLoading ? "Memuat..." : "Muat Ulang Metrik"}
              </button>
            </div>

            {metricsError && (
              <div className="rounded-2xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-800">
                <p className="font-semibold">
                  Tidak dapat mengambil data dari {API_BASE_URL}/metrics
                </p>
                <p className="mt-1 text-xs">{metricsError}</p>
                <p className="mt-2 text-xs text-rose-700">
                  Pastikan backend FastAPI sudah berjalan, CORS mengizinkan
                  domain Vercel Anda, dan environment variable{" "}
                  <code className="rounded bg-rose-100 px-1 py-0.5">
                    NEXT_PUBLIC_API_URL
                  </code>{" "}
                  sudah diatur.
                </p>
              </div>
            )}

            {/* Kartu Model Terpilih */}
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              {/* Jalur A */}
              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <span className="rounded-md bg-indigo-50 px-2.5 py-1 text-xs font-semibold text-indigo-700">
                  Jalur A • Penyaringan Akun Bot
                </span>
                <h3 className="mt-3 text-base font-bold text-slate-900">
                  {metricsData?.jalur_a_selected?.model_name ||
                    "IsolationForest (300 trees, 0.05) + K-Means (k=3)"}
                </h3>
                <dl className="mt-4 grid grid-cols-2 gap-4 text-sm">
                  <div className="rounded-xl bg-slate-50 p-3">
                    <dt className="text-xs text-slate-500">Silhouette Score</dt>
                    <dd className="mt-1 text-lg font-bold text-slate-900">
                      {formatNumber(
                        metricsData?.jalur_a_selected?.silhouette_score,
                        4
                      )}
                    </dd>
                  </div>
                  <div className="rounded-xl bg-slate-50 p-3">
                    <dt className="text-xs text-slate-500">
                      Davies-Bouldin Index
                    </dt>
                    <dd className="mt-1 text-lg font-bold text-slate-900">
                      {formatNumber(
                        metricsData?.jalur_a_selected?.davies_bouldin_index,
                        4
                      )}
                    </dd>
                  </div>
                  <div className="rounded-xl bg-slate-50 p-3">
                    <dt className="text-xs text-slate-500">
                      ARI Stability (IsoForest)
                    </dt>
                    <dd className="mt-1 text-lg font-bold text-emerald-700">
                      {formatNumber(
                        metricsData?.jalur_a_selected?.ari_iso_stability,
                        4
                      )}
                    </dd>
                  </div>
                  <div className="rounded-xl bg-slate-50 p-3">
                    <dt className="text-xs text-slate-500">
                      ARI Stability (K-Means)
                    </dt>
                    <dd className="mt-1 text-lg font-bold text-emerald-700">
                      {formatNumber(
                        metricsData?.jalur_a_selected?.ari_kmeans_stability,
                        4
                      )}
                    </dd>
                  </div>
                </dl>
              </section>

              {/* Jalur B */}
              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <span className="rounded-md bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
                  Jalur B • Klasifikasi Sentimen
                </span>
                <h3 className="mt-3 text-base font-bold text-slate-900">
                  {metricsData?.jalur_b_selected?.model_name ||
                    "Model Terbaik dari Benchmark 5 Algoritma"}
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  Target Macro F1 ≥ {formatPercent(targetMacroF1)} • Confidence
                  Threshold: {activeThreshold.toFixed(2)}
                </p>

                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full divide-y divide-slate-200 text-xs">
                    <thead>
                      <tr className="text-left text-slate-500">
                        <th className="py-2 font-semibold">Metrik</th>
                        <th className="py-2 text-right font-semibold">
                          Split 80/20
                        </th>
                        <th className="py-2 text-right font-semibold">
                          Split 70/30
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      <tr>
                        <td className="py-2.5 font-medium text-slate-700">
                          Macro F1
                        </td>
                        <td className="py-2.5 text-right font-bold text-indigo-700">
                          {formatPercent(
                            split80?.macro_f1 ?? split80?.f1_macro,
                            2
                          )}
                        </td>
                        <td className="py-2.5 text-right font-bold text-indigo-700">
                          {formatPercent(
                            split70?.macro_f1 ?? split70?.f1_macro,
                            2
                          )}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2.5 font-medium text-slate-700">
                          Accuracy
                        </td>
                        <td className="py-2.5 text-right text-slate-900">
                          {formatPercent(split80?.accuracy, 2)}
                        </td>
                        <td className="py-2.5 text-right text-slate-900">
                          {formatPercent(split70?.accuracy, 2)}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2.5 font-medium text-slate-700">
                          Recall Kelas Negatif / Kontra
                        </td>
                        <td className="py-2.5 text-right text-slate-900">
                          {formatPercent(
                            split80?.recall_negatif ?? split80?.recall_negative,
                            2
                          )}
                        </td>
                        <td className="py-2.5 text-right text-slate-900">
                          {formatPercent(
                            split70?.recall_negatif ?? split70?.recall_negative,
                            2
                          )}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </section>
            </div>

            {/* Raw JSON Inspector untuk transparansi TA */}
            {metricsData && (
              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <h3 className="text-sm font-bold text-slate-900">
                  Detail Lengkap JSON dari Backend
                </h3>
                <pre className="mt-3 max-h-72 overflow-auto rounded-xl bg-slate-900 p-4 text-xs text-slate-100">
                  {JSON.stringify(metricsData, null, 2)}
                </pre>
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  );
}