"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";

/**
 * Frontend Next.js (App Router) — Siap Deploy ke Vercel
 * Desain diselaraskan dengan Sistem Pendukung Keputusan AM552 Data Science of IS, FTI UKSW
 *
 * Cara pasang di proyek Next.js (Tailwind CSS aktif):
 *   1. Simpan sebagai `src/app/page.tsx` (atau `app/page.tsx`).
 *   2. Atur Environment Variable di `.env.local` dan Vercel Dashboard:
 *        NEXT_PUBLIC_API_URL=https://<nama-backend-anda>.onrender.com
 */

const RAW_API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_BASE_URL = RAW_API_URL.replace(/\/+$/, "");

// ============================================================================
// TIPE DATA
// ============================================================================
type ActiveTab = "jalur_b" | "jalur_a" | "integrasi";
type SampleKey = "Positif" | "Negatif" | "Netral";

interface MetricsPayload {
  jalur_a_selected?: string;
  jalur_b_selected?: string;
  confidence_threshold?: number;
  jalur_b_cv_macro_f1_80_20?: number;
  jalur_b_test_macro_f1_80_20?: number;
  jalur_b_test_accuracy_80_20?: number;
  jalur_b_test_recall_negatif_80_20?: number;
  jalur_b_test_macro_f1_70_30?: number;
}

interface SentimentResponse {
  sentiment: string;
  confidence: number;
  confidence_threshold: number;
  needs_manual_review: boolean;
  status_operasional: string;
  cleaned_text: string;
  detected_keywords: string[];
  probabilities: Record<string, number>;
  model_name: string;
}

interface BatchSentimentRow {
  Platform: string;
  Komentar: string;
  Sentimen: string;
  Confidence: number;
  Status: string;
}

interface BatchSentimentResponse {
  total_processed: number;
  distribution: Record<string, number>;
  rows: BatchSentimentRow[];
}

interface ScreenedAccount {
  Account_ID: string;
  Platform: string;
  Username: string;
  jumlah_komentar: number;
  jumlah_source_unik: number;
  frekuensi_per_jam: number;
  rasio_duplikasi: number;
  keragaman_teks: number;
  skor_if: number | null;
  kmeans_cluster: number | null;
  Status_Kecukupan: string;
  Status_Akun: string;
}

interface ScreenAccountsResponse {
  summary: {
    total_accounts: number;
    insufficient_data_accounts: number;
    normal_accounts: number;
    flagged_bot_accounts: number;
  };
  accounts: ScreenedAccount[];
}

interface SimulateAccountResponse {
  account: ScreenedAccount | null;
  note: string;
}

interface IntegrationResponse {
  total_comments_before: number;
  total_comments_after: number;
  filtered_out_comments: number;
  comparison: Array<{
    sentimen: string;
    sebelum_pct: number;
    sesudah_pct: number;
    pergeseran_poin: number;
  }>;
  by_category: Array<{
    kategori: string;
    jumlah_komentar: number;
    positif_pct: number;
    negatif_pct: number;
    netral_pct: number;
  }>;
}

// ============================================================================
// DATA REFERENSI CONTOH (TIDAK DAPAT DIHAPUS)
// ============================================================================
const REFERENCE_SAMPLES: Record<SampleKey, string> = {
  Positif: "Tetap semangat mas Gibran, maju terus bekerja nyata untuk rakyat Indonesia.",
  Negatif: "gibran mundur saja",
  Netral: "Siapa kandidat pengganti wakil presiden yang dibahas di podcast ini?",
};

// ============================================================================
// KOMPONEN DONUT CHART SVG (ZERO-DEPENDENCY UNTUK VERCEL BUILD)
// ============================================================================
interface DonutSlice {
  label: string;
  value: number;
  color: string;
}

function DonutChart({
  title,
  slices,
}: {
  title: string;
  slices: DonutSlice[];
}) {
  const total = slices.reduce((sum, s) => sum + Math.max(0, s.value), 0);
  const radius = 64;
  const strokeWidth = 34;
  const circumference = 2 * Math.PI * radius;

  let cumulativePct = 0;

  return (
    <div className="flex flex-col items-center">
      <h4 className="mb-3 text-center text-sm font-semibold text-[#163C66]">
        {title}
      </h4>
      <div className="relative h-52 w-52">
        <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
          {total <= 0 ? (
            <circle
              cx="100"
              cy="100"
              r={radius}
              fill="transparent"
              stroke="#E2E8F0"
              strokeWidth={strokeWidth}
            />
          ) : (
            slices.map((slice) => {
              const pct = Math.max(0, slice.value) / total;
              const dashLength = pct * circumference;
              const dashOffset = -cumulativePct * circumference;
              cumulativePct += pct;
              return (
                <circle
                  key={slice.label}
                  cx="100"
                  cy="100"
                  r={radius}
                  fill="transparent"
                  stroke={slice.color}
                  strokeWidth={strokeWidth}
                  strokeDasharray={`${dashLength} ${circumference}`}
                  strokeDashoffset={dashOffset}
                />
              );
            })
          )}
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap items-center justify-center gap-x-4 gap-y-1.5 text-xs text-slate-700">
        {slices.map((slice) => {
          const pct = total > 0 ? (Math.max(0, slice.value) / total) * 100 : 0;
          return (
            <span key={slice.label} className="inline-flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-2.5 rounded-xs"
                style={{ backgroundColor: slice.color }}
              />
              <span>
                {slice.label}{" "}
                <strong className="font-semibold text-slate-900">
                  ({pct.toFixed(1)}%)
                </strong>
              </span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================================
// HELPER STYLING & EKSPOR CSV
// ============================================================================
function getSentimentPillClass(label: string): string {
  const norm = label.toLowerCase();
  if (norm.includes("positif")) {
    return "bg-[#2ECC71] text-white";
  }
  if (norm.includes("negatif")) {
    return "bg-[#E53935] text-white";
  }
  return "bg-[#2196F3] text-white";
}

function getResultBoxClass(label: string): string {
  const norm = label.toLowerCase();
  if (norm.includes("positif")) {
    return "border-emerald-200 bg-emerald-50/70";
  }
  if (norm.includes("negatif")) {
    return "border-rose-200 bg-rose-50/70";
  }
  return "border-sky-200 bg-sky-50/70";
}

function downloadCsv(filename: string, headers: string[], rows: Array<Array<unknown>>): void {
  const esc = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csv = [headers.join(","), ...rows.map((r) => r.map(esc).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ============================================================================
// KOMPONEN HALAMAN UTAMA
// ============================================================================
export default function OpiniPublikSpkPage() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("jalur_b");
  const [threshold, setThreshold] = useState<number>(0.45);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  // --- Tab 1: Jalur B (Sentimen) ---
  const [refTab, setRefTab] = useState<SampleKey>("Netral");
  const [commentInput, setCommentInput] = useState<string>("gibran mundur saja");
  const [sentLoading, setSentLoading] = useState<boolean>(false);
  const [sentError, setSentError] = useState<string | null>(null);
  const [sentResult, setSentResult] = useState<SentimentResponse | null>(null);

  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchLoading, setBatchLoading] = useState<boolean>(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<BatchSentimentResponse | null>(null);

  // --- Tab 2: Jalur A (Bot Screening) ---
  const [botFile, setBotFile] = useState<File | null>(null);
  const [botLoading, setBotLoading] = useState<boolean>(false);
  const [botError, setBotError] = useState<string | null>(null);
  const [botResult, setBotResult] = useState<ScreenAccountsResponse | null>(null);
  const [onlyFlagged, setOnlyFlagged] = useState<boolean>(true);

  const [simPlatform, setSimPlatform] = useState<string>("YouTube");
  const [simUsername, setSimUsername] = useState<string>("@akun_uji_coba");
  const [simSources, setSimSources] = useState<number>(1);
  const [simComments, setSimComments] = useState<string>(
    "Hoax ini berita tidak benar\nHoax ini berita tidak benar\nHoax ini berita tidak benar\nHoax ini berita tidak benar"
  );
  const [simLoading, setSimLoading] = useState<boolean>(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [simResult, setSimResult] = useState<SimulateAccountResponse | null>(null);

  // --- Tab 3: Integrasi ---
  const [intLoading, setIntLoading] = useState<boolean>(false);
  const [intError, setIntError] = useState<string | null>(null);
  const [intResult, setIntResult] = useState<IntegrationResponse | null>(null);

  // Muat metrik header saat halaman dibuka
  const loadMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/metrics`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: MetricsPayload = await res.json();
      setMetrics(data);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    }
  }, []);

  useEffect(() => {
    void loadMetrics();
  }, [loadMetrics]);

  // Handler Tab 1: Prediksi 1 Komentar
  const handlePredictSingle = async () => {
    const trimmed = commentInput.trim();
    if (!trimmed) {
      setSentError("Masukkan teks komentar terlebih dahulu.");
      return;
    }
    setSentLoading(true);
    setSentError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/predict-sentiment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed, threshold }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: SentimentResponse = await res.json();
      setSentResult(data);
      setApiOnline(true);
    } catch (err) {
      setSentError(err instanceof Error ? err.message : "Gagal menghubungi backend FastAPI.");
    } finally {
      setSentLoading(false);
    }
  };

  // Handler Tab 1: Proses CSV Massal / Demo Proyek
  const handleBatchCsv = async () => {
    if (!batchFile) {
      setBatchError("Pilih file CSV terlebih dahulu.");
      return;
    }
    setBatchLoading(true);
    setBatchError(null);
    try {
      const fd = new FormData();
      fd.append("file", batchFile);
      const res = await fetch(`${API_BASE_URL}/predict-batch?threshold=${threshold}`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());
      setBatchResult(await res.json());
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "Gagal memproses CSV massal.");
    } finally {
      setBatchLoading(false);
    }
  };

  const handleBatchDemo = async () => {
    setBatchLoading(true);
    setBatchError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/predict-demo?threshold=${threshold}&limit=493`);
      if (!res.ok) throw new Error(await res.text());
      setBatchResult(await res.json());
    } catch (err) {
      setBatchError(err instanceof Error ? err.message : "Gagal memuat sampel demo proyek.");
    } finally {
      setBatchLoading(false);
    }
  };

  // Handler Tab 2: Analisis CSV Akun / Demo 1-Klik
  const handleScreenCsv = async () => {
    if (!botFile) {
      setBotError("Pilih file CSV riwayat akun terlebih dahulu.");
      return;
    }
    setBotLoading(true);
    setBotError(null);
    try {
      const fd = new FormData();
      fd.append("file", botFile);
      const res = await fetch(`${API_BASE_URL}/screen-accounts`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());
      setBotResult(await res.json());
    } catch (err) {
      setBotError(err instanceof Error ? err.message : "Gagal menganalisis CSV akun.");
    } finally {
      setBotLoading(false);
    }
  };

  const handleScreenDemo = async () => {
    setBotLoading(true);
    setBotError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/screen-demo`);
      if (!res.ok) throw new Error(await res.text());
      setBotResult(await res.json());
    } catch (err) {
      setBotError(err instanceof Error ? err.message : "Gagal memuat demo dataset proyek.");
    } finally {
      setBotLoading(false);
    }
  };

  // Handler Tab 2: Mode 3 Simulasi Uji 1 Akun
  const handleSimulateAccount = async () => {
    const lines = simComments
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      setSimError("Isi minimal 1 baris komentar untuk simulasi.");
      return;
    }
    setSimLoading(true);
    setSimError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/simulate-account`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          platform: simPlatform,
          username: simUsername.trim() || "@akun_uji_coba",
          n_sources: Math.max(1, simSources),
          comments: lines,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      setSimResult(await res.json());
    } catch (err) {
      setSimError(err instanceof Error ? err.message : "Gagal menjalankan simulasi akun.");
    } finally {
      setSimLoading(false);
    }
  };

  // Handler Tab 3: Integrasi Sebelum vs Sesudah Filter Bot
  const handleLoadIntegration = async () => {
    setIntLoading(true);
    setIntError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/integration-summary`);
      if (!res.ok) throw new Error(await res.text());
      setIntResult(await res.json());
    } catch (err) {
      setIntError(err instanceof Error ? err.message : "Gagal memuat integrasi Jalur A × Jalur B.");
    } finally {
      setIntLoading(false);
    }
  };

  // Data Donut Probabilitas Sentimen
  const sentimentSlices: DonutSlice[] = useMemo(() => {
    if (!sentResult?.probabilities) {
      return [
        { label: "Negatif", value: 0, color: "#E53935" },
        { label: "Netral", value: 0, color: "#2196F3" },
        { label: "Positif", value: 0, color: "#2ECC71" },
      ];
    }
    const p = sentResult.probabilities;
    return [
      { label: "Negatif", value: p["Negatif"] ?? 0, color: "#E53935" },
      { label: "Netral", value: p["Netral"] ?? 0, color: "#2196F3" },
      { label: "Positif", value: p["Positif"] ?? 0, color: "#2ECC71" },
    ];
  }, [sentResult]);

  // Data Donut Segmentasi Akun
  const accountSlices: DonutSlice[] = useMemo(() => {
    if (!botResult) return [];
    return [
      {
        label: "Data Tidak Cukup (1 Komentar)",
        value: botResult.summary.insufficient_data_accounts,
        color: "#95A5A6",
      },
      {
        label: "Aktif Normal (>1 Komentar)",
        value: botResult.summary.normal_accounts,
        color: "#2E86C1",
      },
      {
        label: "Terindikasi Bot / Anomali",
        value: botResult.summary.flagged_bot_accounts,
        color: "#C0392B",
      },
    ];
  }, [botResult]);

  // Tabel akun terfilter
  const displayedAccounts = useMemo(() => {
    if (!botResult) return [];
    if (!onlyFlagged) return botResult.accounts;
    return botResult.accounts.filter(
      (a) => a.Status_Akun === "Terindikasi Bot / Anomali"
    );
  }, [botResult, onlyFlagged]);

  const headerModelText = metrics?.jalur_b_selected
    ? `Aktif • ${metrics.jalur_b_selected}${
        metrics.jalur_b_test_macro_f1_80_20
          ? ` (Macro F1: ${metrics.jalur_b_test_macro_f1_80_20.toFixed(4)})`
          : ""
      }`
    : apiOnline === false
      ? "Backend Belum Terhubung"
      : "Memuat Model...";

  // Status operasional live mengikuti slider threshold
  const liveNeedsReview = sentResult
    ? sentResult.confidence < threshold
    : false;

  return (
    <div className="min-h-screen bg-[#F4F7FB] text-slate-800">
      {/* ==================================================================== */}
      {/* HEADER BIRU TUA (NAVY #163C66)                                       */}
      {/* ==================================================================== */}
      <header className="bg-[#163C66] text-white shadow-sm">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-4 sm:px-6 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-lg font-bold tracking-tight sm:text-xl">
              Analisis Sentimen Opini Publik — Gibran Rakabuming Raka
            </h1>
            <p className="mt-0.5 text-xs text-blue-100/90">
              Sistem Pendukung Keputusan dengan Penyaringan Akun Terindikasi Bot
              • AM552 Data Science of IS, FTI UKSW
            </p>
          </div>
          <div className="inline-flex w-fit items-center gap-2 rounded-full border border-white/25 bg-white/10 px-4 py-1.5 text-xs font-medium text-white">
            <span
              className={`h-2 w-2 rounded-full ${
                apiOnline === true
                  ? "bg-emerald-400"
                  : apiOnline === false
                    ? "bg-rose-400"
                    : "bg-amber-300"
              }`}
            />
            <span>{headerModelText}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-5 px-4 py-5 sm:px-6">
        {/* ================================================================== */}
        {/* KARTU NAVIGASI 3 TAB + SLIDER THRESHOLD KEYAKINAN                  */}
        {/* ================================================================== */}
        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-xs">
          <div className="flex flex-wrap gap-2">
            {[
              { id: "jalur_b" as ActiveTab, label: "1. Analisis Sentimen (Jalur B)" },
              { id: "jalur_a" as ActiveTab, label: "2. Penyaringan Akun Bot (Jalur A)" },
              {
                id: "integrasi" as ActiveTab,
                label: "3. Integrasi: Sebelum vs Sesudah Filter Bot",
              },
            ].map((tab) => {
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`rounded-lg border px-4 py-2 text-xs font-semibold transition ${
                    active
                      ? "border-[#163C66] bg-[#163C66] text-white shadow-xs"
                      : "border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100"
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-slate-100 pt-3">
            <label
              htmlFor="threshold-slider"
              className="text-xs font-medium text-slate-700"
            >
              Threshold Keyakinan:{" "}
              <strong className="font-bold text-slate-900">
                {threshold.toFixed(2)}
              </strong>
            </label>
            <input
              id="threshold-slider"
              type="range"
              min={0.3}
              max={0.9}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="h-1.5 w-44 cursor-pointer accent-[#163C66]"
            />
          </div>
        </section>

        {/* ================================================================== */}
        {/* TAB 1: ANALISIS SENTIMEN (JALUR B)                                 */}
        {/* ================================================================== */}
        {activeTab === "jalur_b" && (
          <div className="space-y-5">
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              {/* Kiri: Uji 1 Komentar (Real-Time) */}
              <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
                <h2 className="text-base font-bold text-[#163C66]">
                  Uji 1 Komentar (Real-Time)
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  Gunakan kotak referensi contoh di atas sebagai panduan, lalu
                  ketik atau tempel komentar yang ingin dianalisis pada kotak
                  input di bawah.
                </p>

                {/* 1. Kotak Referensi Contoh */}
                <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50/70 p-3.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-bold text-slate-800">
                      1. Kotak Referensi Contoh (Tidak Dapat Dihapus)
                    </span>
                    <button
                      type="button"
                      onClick={() => setCommentInput(REFERENCE_SAMPLES[refTab])}
                      className="rounded-md border border-slate-300 bg-white px-2.5 py-1 text-[11px] font-semibold text-[#163C66] hover:bg-slate-50"
                    >
                      Gunakan di Kotak Input ↓
                    </button>
                  </div>

                  <div className="mt-2.5 flex flex-wrap gap-1.5">
                    {(["Positif", "Negatif", "Netral"] as SampleKey[]).map((key) => (
                      <button
                        key={key}
                        type="button"
                        onClick={() => setRefTab(key)}
                        className={`rounded-md border px-2.5 py-1 text-xs font-semibold transition ${
                          refTab === key
                            ? "border-[#163C66] bg-[#163C66] text-white"
                            : "border-slate-200 bg-white text-slate-600 hover:bg-slate-100"
                        }`}
                      >
                        Contoh {key}
                      </button>
                    ))}
                  </div>

                  <div className="mt-2.5 rounded-lg border border-dashed border-sky-200 bg-sky-50/60 p-3 text-xs italic text-slate-700">
                    {REFERENCE_SAMPLES[refTab]}
                  </div>
                </div>

                {/* 2. Kotak Input Komentar */}
                <div className="mt-4 rounded-xl border border-slate-200 bg-white p-3.5">
                  <label
                    htmlFor="single-comment"
                    className="block text-xs font-bold text-slate-800"
                  >
                    2. Kotak Input Komentar untuk Dianalisis
                  </label>
                  <textarea
                    id="single-comment"
                    rows={3}
                    value={commentInput}
                    onChange={(e) => setCommentInput(e.target.value)}
                    placeholder="Masukkan komentar media sosial (termasuk singkatan seperti yg, dgn, tdk)..."
                    className="mt-2 w-full rounded-lg border border-slate-300 p-2.5 text-xs text-slate-900 focus:border-[#163C66] focus:outline-none"
                  />
                  <div className="mt-3 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => void handlePredictSingle()}
                      disabled={sentLoading}
                      className="rounded-lg bg-[#163C66] px-4 py-2 text-xs font-semibold text-white hover:bg-[#123052] disabled:opacity-60"
                    >
                      {sentLoading ? "Menganalisis..." : "Analisis Sentimen Sekarang"}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setCommentInput("");
                        setSentResult(null);
                        setSentError(null);
                      }}
                      className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                    >
                      Bersihkan Input
                    </button>
                  </div>
                </div>

                {sentError && (
                  <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800">
                    {sentError}
                  </div>
                )}

                {/* Kotak Hasil Prediksi */}
                {sentResult && (
                  <div
                    className={`mt-4 space-y-1.5 rounded-xl border p-3.5 text-xs ${getResultBoxClass(
                      sentResult.sentiment
                    )}`}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-bold text-slate-800">Prediksi Utama:</span>
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${getSentimentPillClass(
                          sentResult.sentiment
                        )}`}
                      >
                        {sentResult.sentiment}
                      </span>
                      <span className="font-semibold text-slate-700">
                        ({(sentResult.confidence * 100).toFixed(2)}%)
                      </span>
                    </div>
                    <p className="text-slate-700">
                      <strong className="font-bold text-slate-800">
                        Status Operasional:
                      </strong>{" "}
                      {liveNeedsReview
                        ? `Perlu Tinjauan Manual (Confidence < ${threshold.toFixed(2)})`
                        : "Keyakinan Memadai"}
                    </p>
                    <p className="text-slate-700">
                      <strong className="font-bold text-slate-800">
                        Hasil Preprocessing:
                      </strong>{" "}
                      <span className="font-mono text-[11px]">
                        {sentResult.cleaned_text || "-"}
                      </span>
                    </p>
                    <p className="text-slate-700">
                      <strong className="font-bold text-slate-800">
                        Kata Kunci Terdeteksi:
                      </strong>{" "}
                      {sentResult.detected_keywords?.length
                        ? sentResult.detected_keywords.join(", ")
                        : "-"}
                    </p>
                  </div>
                )}
              </section>

              {/* Kanan: Diagram Probabilitas Sentimen (%) */}
              <section className="flex flex-col justify-between rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
                <div>
                  <h2 className="text-base font-bold text-[#163C66]">
                    Diagram Probabilitas Sentimen (%)
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">
                    Distribusi persentase keyakinan model terhadap kelas Positif,
                    Negatif, dan Netral.
                  </p>
                </div>

                <div className="my-6 flex flex-1 items-center justify-center rounded-xl border border-slate-100 bg-slate-50/40 p-4">
                  <DonutChart
                    title="Probabilitas Sentimen Komentar (%)"
                    slices={sentimentSlices}
                  />
                </div>
              </section>
            </div>

            {/* Bawah: Analisis Sentimen Massal (Dataset CSV) */}
            <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
              <h2 className="text-base font-bold text-[#163C66]">
                Analisis Sentimen Massal (Dataset CSV)
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Unggah file CSV (wajib memiliki kolom Komentar) atau jalankan
                langsung sampel demo proyek (±493 baris).
              </p>

              <div className="mt-3 flex flex-wrap items-center gap-2.5">
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
                  className="text-xs text-slate-600 file:mr-2.5 file:rounded-md file:border file:border-slate-300 file:bg-slate-50 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-slate-700 hover:file:bg-slate-100"
                />
                <button
                  type="button"
                  onClick={() => void handleBatchCsv()}
                  disabled={batchLoading}
                  className="rounded-lg bg-[#243B53] px-4 py-2 text-xs font-semibold text-white hover:bg-[#163C66] disabled:opacity-60"
                >
                  Proses CSV
                </button>
                <button
                  type="button"
                  onClick={() => void handleBatchDemo()}
                  disabled={batchLoading}
                  className="rounded-lg bg-[#163C66] px-4 py-2 text-xs font-semibold text-white hover:bg-[#123052] disabled:opacity-60"
                >
                  {batchLoading ? "Memproses..." : "Muat Demo Proyek"}
                </button>
                {batchResult && (
                  <button
                    type="button"
                    onClick={() =>
                      downloadCsv(
                        "hasil_sentimen_massal.csv",
                        ["Platform", "Komentar", "Sentimen", "Confidence", "Status"],
                        batchResult.rows.map((r) => [
                          r.Platform,
                          r.Komentar,
                          r.Sentimen,
                          r.Confidence,
                          r.Status,
                        ])
                      )
                    }
                    className="rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                  >
                    Unduh Hasil CSV
                  </button>
                )}
              </div>

              {batchError && (
                <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800">
                  {batchError}
                </div>
              )}

              <div className="mt-4 overflow-x-auto rounded-lg border border-slate-200">
                <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                  <thead className="bg-slate-50 text-slate-600">
                    <tr>
                      <th className="px-3 py-2.5 font-semibold">Platform</th>
                      <th className="px-3 py-2.5 font-semibold">Komentar</th>
                      <th className="px-3 py-2.5 font-semibold">Sentimen</th>
                      <th className="px-3 py-2.5 text-right font-semibold">
                        Confidence
                      </th>
                      <th className="px-3 py-2.5 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {!batchResult ? (
                      <tr>
                        <td
                          colSpan={5}
                          className="px-3 py-6 text-center text-slate-400"
                        >
                          Belum ada data analisis massal.
                        </td>
                      </tr>
                    ) : (
                      batchResult.rows.slice(0, 100).map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="px-3 py-2 font-medium text-slate-700">
                            {row.Platform}
                          </td>
                          <td className="max-w-md truncate px-3 py-2 text-slate-700">
                            {row.Komentar}
                          </td>
                          <td className="px-3 py-2">
                            <span
                              className={`rounded-full px-2 py-0.5 text-[11px] font-bold ${getSentimentPillClass(
                                row.Sentimen
                              )}`}
                            >
                              {row.Sentimen}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-slate-700">
                            {(row.Confidence * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-2 text-slate-600">{row.Status}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 2: PENYARINGAN AKUN BOT (JALUR A)                              */}
        {/* ================================================================== */}
        {activeTab === "jalur_a" && (
          <div className="space-y-5">
            {/* Catatan Metodologis */}
            <div className="rounded-xl border border-sky-200 bg-[#EDF5FC] px-4 py-3 text-xs leading-relaxed text-[#163C66]">
              <strong>Catatan Metodologis:</strong> Penyaringan bot mengevaluasi
              pola perilaku multi-komentar pada unit{" "}
              <code className="rounded bg-white/80 px-1 py-0.5 font-mono text-[11px]">
                Platform + Username
              </code>{" "}
              (bukan dari 1 kalimat tunggal). Akun berkomentar tunggal otomatis
              berstatus <em>Data Tidak Cukup</em>, dan hasil anomali bersifat{" "}
              <strong>indikasi awal untuk audit manual</strong>.
            </div>

            {/* Mode 1 & 2 vs Mode 3 */}
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              {/* Mode 1 & 2 */}
              <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
                <h2 className="text-sm font-bold text-[#163C66]">
                  Mode 1 &amp; 2: Demo Proyek / Upload CSV Riwayat Akun
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  Unggah CSV hasil scraping (Platform, Source_ID, Tanggal,
                  Username, Komentar) atau gunakan data demo proyek.
                </p>

                <div className="mt-4 flex flex-wrap items-center gap-2">
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    onChange={(e) => setBotFile(e.target.files?.[0] || null)}
                    className="max-w-[230px] text-xs text-slate-600 file:mr-2 file:rounded-md file:border file:border-slate-300 file:bg-slate-50 file:px-2.5 file:py-1 file:text-xs file:font-semibold"
                  />
                  <button
                    type="button"
                    onClick={() => void handleScreenCsv()}
                    disabled={botLoading}
                    className="rounded-lg bg-[#243B53] px-3.5 py-2 text-xs font-semibold text-white hover:bg-[#163C66] disabled:opacity-60"
                  >
                    Analisis CSV
                  </button>
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void handleScreenDemo()}
                    disabled={botLoading}
                    className="rounded-lg bg-[#163C66] px-4 py-2 text-xs font-semibold text-white hover:bg-[#123052] disabled:opacity-60"
                  >
                    {botLoading ? "Memproses..." : "Demo 1-Klik Dataset Proyek"}
                  </button>
                  <button
                    type="button"
                    disabled={!botResult}
                    onClick={() =>
                      botResult &&
                      downloadCsv(
                        "daftar_prioritas_audit_akun.csv",
                        [
                          "Platform",
                          "Username",
                          "jumlah_komentar",
                          "rasio_duplikasi",
                          "skor_if",
                          "Status_Akun",
                        ],
                        displayedAccounts.map((a) => [
                          a.Platform,
                          a.Username,
                          a.jumlah_komentar,
                          a.rasio_duplikasi,
                          a.skor_if ?? "",
                          a.Status_Akun,
                        ])
                      )
                    }
                    className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                  >
                    Unduh CSV Akun
                  </button>
                </div>

                {botError && (
                  <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800">
                    {botError}
                  </div>
                )}
              </section>

              {/* Mode 3: Simulasi Uji 1 Akun */}
              <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
                <h2 className="text-sm font-bold text-[#163C66]">
                  Mode 3: Simulasi Uji 1 Akun (Multi-Komentar)
                </h2>

                <div className="mt-3 grid grid-cols-3 gap-2">
                  <select
                    value={simPlatform}
                    onChange={(e) => setSimPlatform(e.target.value)}
                    className="rounded-lg border border-slate-300 bg-white px-2.5 py-2 text-xs"
                  >
                    <option value="YouTube">YouTube</option>
                    <option value="Reddit">Reddit</option>
                    <option value="Kaskus">Kaskus</option>
                  </select>
                  <input
                    type="text"
                    value={simUsername}
                    onChange={(e) => setSimUsername(e.target.value)}
                    placeholder="@akun_uji_coba"
                    className="rounded-lg border border-slate-300 px-2.5 py-2 text-xs"
                  />
                  <input
                    type="number"
                    min={1}
                    value={simSources}
                    onChange={(e) => setSimSources(parseInt(e.target.value, 10) || 1)}
                    title="Jumlah Source / Video Unik"
                    className="rounded-lg border border-slate-300 px-2.5 py-2 text-xs"
                  />
                </div>

                <textarea
                  rows={4}
                  value={simComments}
                  onChange={(e) => setSimComments(e.target.value)}
                  placeholder="Tulis 1 komentar per baris..."
                  className="mt-2.5 w-full rounded-lg border border-slate-300 p-2.5 text-xs text-slate-800 focus:border-[#163C66] focus:outline-none"
                />

                <button
                  type="button"
                  onClick={() => void handleSimulateAccount()}
                  disabled={simLoading}
                  className="mt-2.5 rounded-lg bg-[#163C66] px-4 py-2 text-xs font-semibold text-white hover:bg-[#123052] disabled:opacity-60"
                >
                  {simLoading
                    ? "Mengevaluasi..."
                    : "Evaluasi Perilaku Akun Simulasi"}
                </button>

                {simError && (
                  <div className="mt-2.5 rounded-lg border border-rose-200 bg-rose-50 p-2.5 text-xs text-rose-800">
                    {simError}
                  </div>
                )}

                {simResult?.account && (
                  <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="font-bold text-slate-800">
                        {simResult.account.Account_ID}
                      </span>
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold text-white ${
                          simResult.account.Status_Akun.includes("Bot")
                            ? "bg-[#C0392B]"
                            : simResult.account.Status_Akun.includes("Normal")
                              ? "bg-[#2E86C1]"
                              : "bg-slate-500"
                        }`}
                      >
                        {simResult.account.Status_Akun}
                      </span>
                    </div>
                    <p className="mt-1.5 text-slate-600">
                      Komentar: <strong>{simResult.account.jumlah_komentar}</strong>{" "}
                      • Duplikasi:{" "}
                      <strong>
                        {(simResult.account.rasio_duplikasi * 100).toFixed(1)}%
                      </strong>{" "}
                      • Skor IF:{" "}
                      <strong>{simResult.account.skor_if ?? "-"}</strong>
                    </p>
                  </div>
                )}
              </section>
            </div>

            {/* Bawah: Hasil Segmentasi & Daftar Prioritas Audit Akun */}
            <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
              <h2 className="text-sm font-bold text-[#163C66]">
                Hasil Segmentasi &amp; Daftar Prioritas Audit Akun
              </h2>

              {/* 4 KPI Cards */}
              <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
                <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-3.5">
                  <p className="text-[11px] font-medium text-slate-500">
                    Total Akun Unik
                  </p>
                  <p className="mt-1 text-xl font-bold text-[#163C66]">
                    {botResult
                      ? botResult.summary.total_accounts.toLocaleString("en-US")
                      : "-"}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-3.5">
                  <p className="text-[11px] font-medium text-slate-500">
                    1 Komentar (Tidak Cukup)
                  </p>
                  <p className="mt-1 text-xl font-bold text-slate-500">
                    {botResult
                      ? botResult.summary.insufficient_data_accounts.toLocaleString(
                          "en-US"
                        )
                      : "-"}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-3.5">
                  <p className="text-[11px] font-medium text-slate-500">
                    Aktif Normal (&gt;1 Komen)
                  </p>
                  <p className="mt-1 text-xl font-bold text-[#2E86C1]">
                    {botResult
                      ? botResult.summary.normal_accounts.toLocaleString("en-US")
                      : "-"}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-3.5">
                  <p className="text-[11px] font-medium text-slate-500">
                    Terindikasi Bot / Anomali
                  </p>
                  <p className="mt-1 text-xl font-bold text-[#C0392B]">
                    {botResult
                      ? botResult.summary.flagged_bot_accounts.toLocaleString(
                          "en-US"
                        )
                      : "-"}
                  </p>
                </div>
              </div>

              {/* Donut + Tabel Prioritas Audit */}
              <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-12">
                <div className="flex items-center justify-center rounded-xl border border-slate-200 bg-white p-4 lg:col-span-5">
                  <DonutChart
                    title="Segmentasi Status Perilaku Akun"
                    slices={accountSlices}
                  />
                </div>

                <div className="lg:col-span-7">
                  <label className="mb-2 inline-flex cursor-pointer items-center gap-2 text-xs text-slate-700">
                    <input
                      type="checkbox"
                      checked={onlyFlagged}
                      onChange={(e) => setOnlyFlagged(e.target.checked)}
                      className="rounded accent-[#163C66]"
                    />
                    <span>
                      Tampilkan hanya akun{" "}
                      <strong>Terindikasi Bot / Anomali</strong>
                    </span>
                  </label>

                  <div className="max-h-72 overflow-auto rounded-lg border border-slate-200">
                    <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                      <thead className="sticky top-0 bg-slate-50 text-slate-600">
                        <tr>
                          <th className="px-3 py-2.5 font-semibold">Platform</th>
                          <th className="px-3 py-2.5 font-semibold">Username</th>
                          <th className="px-3 py-2.5 text-right font-semibold">
                            Jml Komen
                          </th>
                          <th className="px-3 py-2.5 text-right font-semibold">
                            Duplikasi
                          </th>
                          <th className="px-3 py-2.5 text-right font-semibold">
                            Skor IF
                          </th>
                          <th className="px-3 py-2.5 font-semibold">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white">
                        {displayedAccounts.length === 0 ? (
                          <tr>
                            <td
                              colSpan={6}
                              className="px-3 py-8 text-center text-slate-400"
                            >
                              Klik &ldquo;Demo 1-Klik Dataset Proyek&rdquo; atau
                              unggah CSV untuk melihat daftar audit akun.
                            </td>
                          </tr>
                        ) : (
                          displayedAccounts.slice(0, 200).map((acc) => (
                            <tr
                              key={acc.Account_ID}
                              className="hover:bg-slate-50"
                            >
                              <td className="px-3 py-2 text-slate-700">
                                {acc.Platform}
                              </td>
                              <td className="max-w-[160px] truncate px-3 py-2 font-mono text-slate-800">
                                {acc.Username}
                              </td>
                              <td className="px-3 py-2 text-right font-semibold text-slate-800">
                                {acc.jumlah_komentar}
                              </td>
                              <td className="px-3 py-2 text-right text-slate-700">
                                {(acc.rasio_duplikasi * 100).toFixed(1)}%
                              </td>
                              <td className="px-3 py-2 text-right font-mono text-slate-700">
                                {acc.skor_if !== null ? acc.skor_if.toFixed(4) : "-"}
                              </td>
                              <td className="px-3 py-2">
                                <span
                                  className={`inline-block whitespace-nowrap rounded-full px-2.5 py-0.5 text-[11px] font-bold text-white ${
                                    acc.Status_Akun.includes("Bot")
                                      ? "bg-[#C0392B]"
                                      : acc.Status_Akun.includes("Normal")
                                        ? "bg-[#2E86C1]"
                                        : "bg-slate-400"
                                  }`}
                                >
                                  {acc.Status_Akun}
                                </span>
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 3: INTEGRASI SEBELUM VS SESUDAH FILTER BOT                     */}
        {/* ================================================================== */}
        {activeTab === "integrasi" && (
          <div className="space-y-5">
            <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-bold text-[#163C66]">
                    Integrasi Benang Merah: Sebelum vs Sesudah Penyaringan Bot
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">
                    Mengukur pergeseran distribusi opini publik (Jalur B) setelah
                    komentar dari akun terindikasi bot/anomali (Jalur A)
                    disaring.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void handleLoadIntegration()}
                  disabled={intLoading}
                  className="rounded-lg bg-[#163C66] px-4 py-2 text-xs font-semibold text-white hover:bg-[#123052] disabled:opacity-60"
                >
                  {intLoading
                    ? "Menghitung Integrasi..."
                    : "Jalankan Analisis Integrasi"}
                </button>
              </div>

              {intError && (
                <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800">
                  {intError}
                </div>
              )}

              {intResult && (
                <div className="mt-5 space-y-5">
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                    <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-4">
                      <p className="text-xs text-slate-500">
                        Total Komentar (Sebelum Filter)
                      </p>
                      <p className="mt-1 text-xl font-bold text-[#163C66]">
                        {intResult.total_comments_before.toLocaleString("en-US")}
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-4">
                      <p className="text-xs text-slate-500">
                        Komentar Disaring (Akun Anomali)
                      </p>
                      <p className="mt-1 text-xl font-bold text-[#C0392B]">
                        {intResult.filtered_out_comments.toLocaleString("en-US")}
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-4">
                      <p className="text-xs text-slate-500">
                        Total Komentar (Sesudah Filter)
                      </p>
                      <p className="mt-1 text-xl font-bold text-emerald-700">
                        {intResult.total_comments_after.toLocaleString("en-US")}
                      </p>
                    </div>
                  </div>

                  {/* Tabel Dampak Penyaringan */}
                  <div className="overflow-x-auto rounded-lg border border-slate-200">
                    <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                      <thead className="bg-slate-50 text-slate-600">
                        <tr>
                          <th className="px-4 py-2.5 font-semibold">
                            Kelas Sentimen
                          </th>
                          <th className="px-4 py-2.5 text-right font-semibold">
                            Sebelum Penyaringan (%)
                          </th>
                          <th className="px-4 py-2.5 text-right font-semibold">
                            Sesudah Penyaringan (%)
                          </th>
                          <th className="px-4 py-2.5 text-right font-semibold">
                            Pergeseran (Poin %)
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white">
                        {intResult.comparison.map((row) => (
                          <tr key={row.sentimen}>
                            <td className="px-4 py-2.5 font-semibold text-slate-800">
                              <span
                                className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${getSentimentPillClass(
                                  row.sentimen
                                )}`}
                              >
                                {row.sentimen}
                              </span>
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono">
                              {row.sebelum_pct.toFixed(2)}%
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono font-semibold">
                              {row.sesudah_pct.toFixed(2)}%
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono font-bold text-[#163C66]">
                              {row.pergeseran_poin >= 0 ? "+" : ""}
                              {row.pergeseran_poin.toFixed(2)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Proporsi Sentimen per Kategori Akun */}
                  <div className="overflow-x-auto rounded-lg border border-slate-200">
                    <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                      <thead className="bg-slate-50 text-slate-600">
                        <tr>
                          <th className="px-4 py-2.5 font-semibold">
                            Kategori Akun (Jalur A)
                          </th>
                          <th className="px-4 py-2.5 text-right font-semibold">
                            Jumlah Komentar
                          </th>
                          <th className="px-4 py-2.5 text-right font-semibold">
                            Positif (%)
                          </th>
                          <th className="px-4 py-2.5 text-right font-semibold">
                            Negatif (%)
                          </th>
                          <th className="px-4 py-2.5 text-right font-semibold">
                            Netral (%)
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white">
                        {intResult.by_category.map((cat) => (
                          <tr key={cat.kategori}>
                            <td className="px-4 py-2.5 font-semibold text-slate-800">
                              {cat.kategori}
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono">
                              {cat.jumlah_komentar.toLocaleString("en-US")}
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono text-emerald-700">
                              {cat.positif_pct.toFixed(2)}%
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono text-rose-700">
                              {cat.negatif_pct.toFixed(2)}%
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono text-sky-700">
                              {cat.netral_pct.toFixed(2)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </section>

            {/* Ringkasan Metrik Evaluasi */}
            {metrics && (
              <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
                <h3 className="text-sm font-bold text-[#163C66]">
                  Ringkasan Metrik Evaluasi Model (metrics.json)
                </h3>
                <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <p className="text-[11px] text-slate-500">
                      Test Macro F1 (80/20)
                    </p>
                    <p className="mt-1 text-lg font-bold text-[#163C66]">
                      {metrics.jalur_b_test_macro_f1_80_20?.toFixed(4) ?? "-"}
                    </p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <p className="text-[11px] text-slate-500">
                      Test Accuracy (80/20)
                    </p>
                    <p className="mt-1 text-lg font-bold text-[#163C66]">
                      {metrics.jalur_b_test_accuracy_80_20?.toFixed(4) ?? "-"}
                    </p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <p className="text-[11px] text-slate-500">
                      Recall Negatif (80/20)
                    </p>
                    <p className="mt-1 text-lg font-bold text-rose-700">
                      {metrics.jalur_b_test_recall_negatif_80_20?.toFixed(4) ?? "-"}
                    </p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <p className="text-[11px] text-slate-500">
                      Test Macro F1 (70/30)
                    </p>
                    <p className="mt-1 text-lg font-bold text-[#2E86C1]">
                      {metrics.jalur_b_test_macro_f1_70_30?.toFixed(4) ?? "-"}
                    </p>
                  </div>
                </div>
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  );
}