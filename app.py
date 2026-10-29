from pathlib import Path
from datetime import datetime
import io
import json
import os
import subprocess

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# LOKASI PROYEK
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXE = ROOT / "bin" / "benchmark.exe"

RESULTS.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "D1": 1_000,
    "D2": 10_000,
    "D3": 100_000,
    "D4": 1_000_000,
    "D5": 10_000_000,
    "D6": 100_000_000,
}

WORKERS = [1, 2, 4, 8, 16]

st.set_page_config(
    page_title="Benchmark Min, Max, dan Rata-rata",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background: rgba(110, 130, 160, 0.10);
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MEMBACA DAN MEMERIKSA CSV
# ============================================================

def baca_csv(payload):
    """Menerima bytes CSV asli dari program benchmark C."""
    df = pd.read_csv(io.BytesIO(payload))

    wajib = [
        "dataset_id",
        "dataset_size",
        "mode",
        "workers_requested",
        "workers_actual",
        "iteration",
        "seed",
        "min",
        "max",
        "sum",
        "average",
        "time_seconds",
        "time_ms",
        "timer",
        "elapsed_ticks",
        "timer_frequency_hz",
        "valid",
    ]

    kurang = sorted(set(wajib) - set(df.columns))
    if kurang:
        raise ValueError(
            "Kolom CSV tidak lengkap: " + ", ".join(kurang)
        )

    numerik = [
        kolom for kolom in wajib
        if kolom not in ["dataset_id", "mode", "timer"]
    ]

    for kolom in numerik:
        df[kolom] = pd.to_numeric(df[kolom], errors="raise")

    if not np.isfinite(df[numerik].to_numpy(dtype=float)).all():
        raise ValueError("CSV mengandung angka kosong atau tidak valid.")

    kolom_integer = [
        "dataset_size", "workers_requested", "workers_actual",
        "iteration", "seed", "min", "max", "sum",
        "elapsed_ticks", "timer_frequency_hz", "valid",
    ]

    for kolom in kolom_integer:
        if not (df[kolom] == np.floor(df[kolom])).all():
            raise ValueError(
                f"Kolom {kolom} harus berisi bilangan bulat."
            )
        df[kolom] = df[kolom].astype("int64")

    if len(df) != 150:
        raise ValueError(
            f"CSV berisi {len(df)} baris. "
            "Aplikasi ini memerlukan hasil lengkap 150 pengukuran."
        )

    if not df["dataset_id"].isin(DATASETS).all():
        raise ValueError("Terdapat ID dataset di luar D1-D6.")

    ukuran_benar = df["dataset_id"].map(DATASETS)
    if not df["dataset_size"].eq(ukuran_benar).all():
        raise ValueError("Ukuran dataset tidak cocok dengan ID D1-D6.")

    if not df["workers_requested"].isin(WORKERS).all():
        raise ValueError("Konfigurasi worker tidak sesuai penelitian.")

    if not df["iteration"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Nomor pengulangan harus 1 sampai 5.")

    kunci = ["dataset_id", "workers_requested", "iteration"]
    if df.duplicated(kunci).any():
        raise ValueError("Ada pengukuran dengan kunci yang berulang.")

    kelompok = df.groupby(
        ["dataset_id", "workers_requested"]
    ).size()

    if len(kelompok) != 30 or not kelompok.eq(5).all():
        raise ValueError(
            "Harus tersedia 30 kondisi dengan 5 pengulangan."
        )

    if not df["valid"].eq(1).all():
        raise ValueError("Program C menandai pengukuran tidak valid.")

    if not df["workers_actual"].eq(df["workers_requested"]).all():
        raise ValueError("Jumlah thread aktual berbeda dari permintaan.")

    mode_benar = np.where(
        df["workers_requested"].eq(1),
        "sequential",
        "parallel",
    )

    if not df["mode"].eq(mode_benar).all():
        raise ValueError("Label mode sekuensial/paralel tidak sesuai.")

    if not df["timer"].eq("QPC").all():
        raise ValueError("CSV harus berasal dari versi timer QPC.")

    if not (
        df["elapsed_ticks"].gt(0).all()
        and df["timer_frequency_hz"].gt(0).all()
    ):
        raise ValueError("Tick atau frekuensi timer tidak positif.")

    waktu_detik = (
        df["elapsed_ticks"] / df["timer_frequency_hz"]
    )

    if not np.allclose(
        df["time_seconds"],
        waktu_detik,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(
            "Kolom time_seconds tidak sesuai tick. "
            "Gunakan CSV asli program C, bukan hasil ekspor Excel."
        )

    if not np.allclose(
        df["time_ms"],
        waktu_detik * 1000,
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError(
            "Kolom time_ms tidak sesuai konversi timer."
        )

    if not np.allclose(
        df["average"],
        df["sum"] / df["dataset_size"],
        rtol=1e-12,
        atol=1e-8,
    ):
        raise ValueError("Nilai average tidak sesuai sum / dataset_size.")

    if not (
        df["min"].ge(0).all()
        and df["max"].le(65535).all()
        and df["min"].le(df["max"]).all()
    ):
        raise ValueError("Rentang nilai dataset tidak sesuai generator.")

    konsistensi = df.groupby("dataset_id")[
        ["min", "max", "sum", "average", "seed"]
    ].nunique()

    if not konsistensi.eq(1).all().all():
        raise ValueError(
            "Hasil atau seed berbeda antarworker pada dataset yang sama."
        )

    # Gunakan tick sebagai sumber waktu perhitungan.
    df["time_seconds"] = waktu_detik
    df["time_ms"] = waktu_detik * 1000

    return df


def buat_ringkasan(df):
    ringkas = df.groupby(
        ["dataset_id", "dataset_size", "workers_requested"],
        as_index=False,
    ).agg(
        repetitions=("time_ms", "size"),
        mean_ms=("time_ms", "mean"),
        sd_ms=("time_ms", "std"),
        median_ms=("time_ms", "median"),
        min_ticks=("elapsed_ticks", "min"),
        minimum=("min", "first"),
        maximum=("max", "first"),
        total=("sum", "first"),
        average=("average", "first"),
    )

    baseline = ringkas.loc[
        ringkas["workers_requested"].eq(1),
        ["dataset_id", "mean_ms"],
    ].rename(columns={"mean_ms": "sequential_ms"})

    ringkas = ringkas.merge(
        baseline,
        on="dataset_id",
        validate="many_to_one",
    )

    ringkas["speedup"] = (
        ringkas["sequential_ms"] / ringkas["mean_ms"]
    )

    ringkas["efficiency_pct"] = (
        ringkas["speedup"]
        / ringkas["workers_requested"]
        * 100
    )

    ringkas["cv_pct"] = (
        ringkas["sd_ms"] / ringkas["mean_ms"] * 100
    )

    ringkas["worker_label"] = ringkas[
        "workers_requested"
    ].map(
        lambda w: "1 - Sekuensial" if w == 1 else f"{w} - Paralel"
    )

    return ringkas.sort_values(
        ["dataset_size", "workers_requested"]
    ).reset_index(drop=True)


# ============================================================
# MENJALANKAN PROGRAM C
# ============================================================

def jalankan_benchmark(charger, mode_daya):
    if not EXE.is_file():
        raise FileNotFoundError(
            f"Program C belum ditemukan: {EXE}"
        )

    lock = RESULTS / "benchmark_web.lock"

    try:
        pegangan = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(
            "Pengujian web lain sedang berjalan. "
            "Tunggu sampai selesai."
        ) from None

    try:
        with pegangan:
            pegangan.write(str(os.getpid()))

        waktu = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = RESULTS / f"hasil_web_{waktu}.csv"

        env = os.environ.copy()
        mingw = Path("C:/msys64/mingw64/bin")

        if mingw.is_dir():
            env["PATH"] = str(mingw) + os.pathsep + env.get("PATH", "")

        proses = subprocess.run(
            [str(EXE), str(output)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            shell=False,
        )

        log = proses.stdout + "\n" + proses.stderr
        output.with_suffix(".log").write_text(
            log, encoding="utf-8"
        )

        if proses.returncode != 0:
            raise RuntimeError(
                "Program C berhenti dengan error:\n" + log
            )

        # Hanya menerima hasil lengkap yang lolos pemeriksaan.
        baca_csv(output.read_bytes())

        metadata = {
            "waktu_mulai_lokal": waktu,
            "charger": charger,
            "mode_daya": mode_daya.strip() or "Tidak tercatat",
            "program": str(EXE),
            "jumlah_pengukuran": 150,
            "stdout": proses.stdout,
        }

        output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return output, log

    finally:
        lock.unlink(missing_ok=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Panel Pengujian")
st.sidebar.caption("C + OpenMP · Antarmuka Streamlit")

with st.sidebar.expander("Perangkat pengujian", expanded=False):
    st.write("**ASUS VivoBook M1405YA**")
    st.write("AMD Ryzen 7 7730U")
    st.write("8 core / 16 thread")
    st.write("RAM 16 GB · Windows 11 64-bit")
    st.caption(
        "Spesifikasi berdasarkan informasi penelitian, "
        "bukan deteksi otomatis browser."
    )

st.sidebar.subheader("Pengujian baru")

charger = st.sidebar.selectbox(
    "Status charger saat pengujian",
    ["Tidak tercatat", "Terpasang", "Tidak terpasang"],
)

mode_daya = st.sidebar.text_input(
    "Mode daya Windows",
    placeholder="Misalnya: Balanced, atau Tidak tercatat",
)

st.sidebar.caption(
    "Tombol ini menjalankan D1-D6 dengan seluruh konfigurasi "
    "worker, masing-masing 5 kali. Tutup aplikasi berat "
    "dan hindari menjalankan benchmark lain bersamaan."
)

if st.sidebar.button(
    "Jalankan 150 Pengukuran",
    type="primary",
    disabled=not EXE.is_file(),
):
    try:
        with st.spinner("Program C sedang melakukan pengujian..."):
            output, log = jalankan_benchmark(charger, mode_daya)

        st.session_state["file_hasil"] = str(output)
        st.session_state["sumber_data"] = "Hasil lokal"
        st.session_state["log_terakhir"] = log
        st.success(
            "150 pengukuran selesai. Hasil baru sudah dipilih."
        )

    except Exception as exc:
        st.error(str(exc))

st.sidebar.divider()
st.sidebar.subheader("Sumber hasil")

sumber = st.sidebar.radio(
    "Baca hasil dari",
    ["Hasil lokal", "Unggah CSV"],
    key="sumber_data",
)

payload = None
nama_sumber = ""
path_terpilih = None

if sumber == "Hasil lokal":
    # Fallback pencarian otomatis file default jika belum tersetting
    if not list(RESULTS.glob("*.csv")):
        default_root_csv = ROOT / "hasil_pengujian.csv"
        if default_root_csv.is_file() and not (RESULTS / "hasil_pengujian.csv").is_file():
            import shutil
            shutil.copy(default_root_csv, RESULTS / "hasil_pengujian.csv")

    daftar = sorted(
        RESULTS.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    pilihan = [str(p) for p in daftar]

    if pilihan:
        if st.session_state.get("file_hasil") not in pilihan:
            asli = RESULTS / "hasil_pengujian.csv"
            st.session_state["file_hasil"] = (
                str(asli) if str(asli) in pilihan else pilihan[0]
            )

        terpilih = st.sidebar.selectbox(
            "File hasil",
            pilihan,
            format_func=lambda p: Path(p).name,
            key="file_hasil",
        )

        path_terpilih = Path(terpilih)
        payload = path_terpilih.read_bytes()
        nama_sumber = path_terpilih.name
    else:
        st.sidebar.info(
            "Belum ada CSV. Jalankan pengujian atau unggah CSV."
        )

else:
    unggahannya = st.sidebar.file_uploader(
        "Unggah CSV asli keluaran program C",
        type=["csv"],
    )

    if unggahannya is not None:
        payload = unggahannya.getvalue()
        nama_sumber = unggahannya.name


# ============================================================
# HALAMAN UTAMA
# ============================================================

st.title("Benchmark Min, Max, dan Rata-rata")
st.write(
    "Analisis pengaruh ukuran dataset dan jumlah worker "
    "terhadap waktu eksekusi sekuensial dan paralel OpenMP."
)
st.caption(
    "Satu pengukuran mencakup min, max, dan penjumlahan "
    "dalam satu traversal, diikuti perhitungan rata-rata."
)

if payload is None:
    st.info(
        "Pilih hasil lokal atau unggah hasil_pengujian.csv "
        "melalui panel di sebelah kiri."
    )
    st.stop()

try:
    data = baca_csv(payload)
    ringkasan = buat_ringkasan(data)
except Exception as exc:
    st.error(f"CSV belum dapat digunakan: {exc}")
    st.stop()

st.success(
    f"{nama_sumber} · 150 pengukuran · "
    "30 kondisi · 5 pengulangan · pemeriksaan data lulus"
)

with st.expander("Catatan kondisi pengujian"):
    metadata_path = (
        path_terpilih.with_suffix(".json")
        if path_terpilih is not None else None
    )

    if metadata_path is not None and metadata_path.exists():
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            st.write(
                "**Status charger:**",
                metadata.get("charger", "Tidak tercatat"),
            )
            st.write(
                "**Mode daya:**",
                metadata.get("mode_daya", "Tidak tercatat"),
            )
        except (OSError, ValueError):
            st.write("Metadata kondisi pengujian tidak dapat dibaca.")
    else:
        st.write(
            "Status charger dan mode daya tidak tersedia "
            "dalam CSV ini."
        )

st.sidebar.subheader("Filter tampilan")

dataset_pilih = st.sidebar.multiselect(
    "Dataset yang ditampilkan",
    list(DATASETS),
    default=list(DATASETS),
    format_func=lambda d: (
        f"{d} — {DATASETS[d]:,} elemen".replace(",", ".")
    ),
)

worker_pilih = st.sidebar.multiselect(
    "Worker yang ditampilkan",
    WORKERS,
    default=WORKERS,
)

st.sidebar.caption(
    "Filter hanya mengubah tampilan. Baseline speedup tetap "
    "menggunakan seluruh 5 pengukuran sekuensial dataset terkait."
)

if not dataset_pilih or not worker_pilih:
    st.info("Pilih minimal satu dataset dan satu konfigurasi worker.")
    st.stop()

tampil = ringkasan.loc[
    ringkasan["dataset_id"].isin(dataset_pilih)
    & ringkasan["workers_requested"].isin(worker_pilih)
].copy()

mentah_tampil = data.loc[
    data["dataset_id"].isin(dataset_pilih)
    & data["workers_requested"].isin(worker_pilih)
].copy()

c1, c2, c3 = st.columns(3)
c1.metric("Dataset ditampilkan", len(dataset_pilih))
c2.metric("Kondisi ditampilkan", len(tampil))
c3.metric("Pengukuran ditampilkan", len(mentah_tampil))

tab_hasil, tab_grafik, tab_data, tab_metode = st.tabs(
    ["Hasil Pengujian", "Grafik", "Data Mentah", "Metodologi"]
)


# ============================================================
# TAB HASIL
# ============================================================

with tab_hasil:
    st.subheader("Ringkasan performa")

    kolom = {
        "dataset_id": "Dataset",
        "dataset_size": "Jumlah elemen",
        "worker_label": "Worker",
        "repetitions": "Ulangan",
        "mean_ms": "Rata-rata waktu (ms)",
        "sd_ms": "SD sampel (ms)",
        "cv_pct": "CV (%)",
        "speedup": "Speedup (kali)",
        "efficiency_pct": "Efficiency (%)",
    }

    tabel = tampil[list(kolom)].rename(columns=kolom)

    st.dataframe(
        tabel,
        hide_index=True,
        width="stretch",
        column_config={
            "Jumlah elemen": st.column_config.NumberColumn(
                format="%d"
            ),
            "Rata-rata waktu (ms)": st.column_config.NumberColumn(
                format="%.5f"
            ),
            "SD sampel (ms)": st.column_config.NumberColumn(
                format="%.8f"
            ),
            "CV (%)": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "Speedup (kali)": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Efficiency (%)": st.column_config.NumberColumn(
                format="%.2f"
            ),
        },
    )

    st.subheader("Hasil perhitungan dataset")

    hasil_dataset = tampil.drop_duplicates("dataset_id")[
        [
            "dataset_id", "dataset_size", "minimum",
            "maximum", "total", "average",
        ]
    ].rename(
        columns={
            "dataset_id": "Dataset",
            "dataset_size": "Jumlah elemen",
            "minimum": "Minimum",
            "maximum": "Maksimum",
            "total": "Jumlah nilai",
            "average": "Rata-rata nilai",
        }
    )

    st.dataframe(
        hasil_dataset,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Rata-rata nilai adalah hasil operasi pada dataset. "
        "Rata-rata waktu adalah rata-rata durasi lima pengulangan."
    )

    st.download_button(
        "Unduh Ringkasan Tampilan",
        data=tabel.to_csv(index=False).encode("utf-8-sig"),
        file_name="ringkasan_performa.csv",
        mime="text/csv",
        key="unduh_ringkasan",
    )

    st.subheader("Perbandingan pada satu dataset")

    fokus = st.selectbox(
        "Dataset fokus",
        dataset_pilih,
        key="dataset_fokus",
    )

    # Bandingkan seluruh konfigurasi untuk dataset fokus.
    fokus_data = ringkasan.loc[
        ringkasan["dataset_id"].eq(fokus)
    ]

    terbaik = fokus_data.loc[fokus_data["mean_ms"].idxmin()]

    st.info(
        f"Pada {fokus}, konfigurasi dengan rata-rata waktu "
        f"terendah adalah {int(terbaik['workers_requested'])} worker: "
        f"{terbaik['mean_ms']:.5f} ms, "
        f"speedup {terbaik['speedup']:.3f} kali, "
        f"efficiency {terbaik['efficiency_pct']:.2f}%. "
        "Perbandingan ini mencakup seluruh konfigurasi worker."
    )


# ============================================================
# TAB GRAFIK
# ============================================================

with tab_grafik:
    st.subheader("Ukuran dataset dan waktu eksekusi")

    log_waktu = st.checkbox(
        "Gunakan skala logaritmik untuk waktu",
        value=True,
    )

    fig_waktu = px.line(
        tampil,
        x="dataset_size",
        y="mean_ms",
        color="worker_label",
        markers=True,
        log_x=True,
        log_y=log_waktu,
        labels={
            "dataset_size": "Jumlah elemen",
            "mean_ms": "Rata-rata waktu eksekusi (ms)",
            "worker_label": "Worker",
        },
        hover_data={"sd_ms": ":.8f", "cv_pct": ":.2f"},
    )

    fig_waktu.update_xaxes(
        tickvals=list(DATASETS.values()),
        ticktext=[
            f"{d}: {n:,}".replace(",", ".")
            for d, n in DATASETS.items()
        ],
    )

    st.plotly_chart(
        fig_waktu,
        width="stretch",
        key="grafik_waktu",
    )

    st.subheader("Jumlah worker dan speedup")

    fig_speedup = px.line(
        tampil,
        x="workers_requested",
        y="speedup",
        color="dataset_id",
        markers=True,
        labels={
            "workers_requested": "Jumlah worker (thread)",
            "speedup": "Speedup (kali)",
            "dataset_id": "Dataset",
        },
    )

    fig_speedup.update_xaxes(
        type="linear", tickvals=WORKERS
    )
    fig_speedup.add_hline(
        y=1,
        line_dash="dash",
        annotation_text="S = 1: waktu sama dengan sekuensial",
    )

    st.plotly_chart(
        fig_speedup,
        width="stretch",
        key="grafik_speedup",
    )

    st.subheader("Jumlah worker dan efficiency")

    fig_efficiency = px.line(
        tampil,
        x="workers_requested",
        y="efficiency_pct",
        color="dataset_id",
        markers=True,
        labels={
            "workers_requested": "Jumlah worker (thread)",
            "efficiency_pct": "Efficiency (%)",
            "dataset_id": "Dataset",
        },
    )

    fig_efficiency.update_xaxes(
        type="linear", tickvals=WORKERS
    )

    st.plotly_chart(
        fig_efficiency,
        width="stretch",
        key="grafik_efficiency",
    )

    st.subheader("Variasi waktu antarulangan")

    fig_variasi = px.box(
        mentah_tampil,
        x="workers_requested",
        y="time_ms",
        color="dataset_id",
        points="all",
        log_y=True,
        labels={
            "workers_requested": "Jumlah worker (thread)",
            "time_ms": "Waktu eksekusi (ms, skala logaritmik)",
            "dataset_id": "Dataset",
        },
    )

    st.plotly_chart(
        fig_variasi,
        width="stretch",
        key="grafik_variasi",
    )

    st.caption(
        "Setiap kondisi memiliki 5 pengukuran. Seluruh titik "
        "ditampilkan, termasuk pengukuran yang lebih lambat."
    )


# ============================================================
# TAB DATA MENTAH
# ============================================================

with tab_data:
    st.subheader("Pengukuran individual")

    st.dataframe(
        mentah_tampil.sort_values(
            ["dataset_size", "workers_requested", "iteration"]
        ),
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Unduh CSV Asli Lengkap",
        data=payload,
        file_name=Path(nama_sumber).name,
        mime="text/csv",
        key="unduh_asli",
    )

    st.caption(
        "CSV asli lengkap tetap berisi 150 pengukuran, "
        "meskipun filter tampilan sedang digunakan. "
        "Saat dibuka di Excel, gunakan pemisah desimal titik "
        "agar angka tidak salah terbaca."
    )


# ============================================================
# TAB METODOLOGI
# ============================================================

with tab_metode:
    st.subheader("Rancangan pengujian")

    st.markdown(
        """
        - Enam ukuran dataset: 1.000 sampai 100.000.000 elemen.
        - Dataset berupa bilangan bulat 32-bit dalam rentang 0–65.535.
        - Data dibuat menggunakan generator pseudoacak dengan seed tetap.
        - Baseline memakai loop sekuensial biasa dengan 1 worker.
        - Paralel memakai OpenMP dengan 2, 4, 8, dan 16 thread.
        - Min, max, dan jumlah dihitung dalam satu traversal.
        - Rata-rata nilai dihitung dari jumlah dibagi banyak elemen.
        - Setiap kondisi diukur 5 kali, menghasilkan 150 pengukuran.
        - Timer pengukuran adalah QueryPerformanceCounter.
        - Pembuatan dataset, pemanasan, validasi, dan penulisan CSV
          berada di luar pengukuran waktu.
        - Region paralel dan reduction berada di dalam pengukuran.
        - Urutan konfigurasi worker digilir pada setiap pengulangan.
        """
    )

    st.subheader("Rumus evaluasi")

    st.latex(r"\bar{T}=\frac{1}{5}\sum_{i=1}^{5}T_i")
    st.latex(r"S_p=\frac{\bar{T}_s}{\bar{T}_p}")
    st.latex(r"E_p=\frac{S_p}{p}\times100\%")
    st.latex(
        r"s=\sqrt{\frac{\sum_{i=1}^{5}(T_i-\bar{T})^2}{5-1}}"
    )
    st.latex(r"CV=\frac{s}{\bar{T}}\times100\%")

    st.write(
        "Speedup dihitung dari perbandingan rata-rata waktu, "
        "bukan rata-rata dari lima rasio speedup."
    )

    st.subheader("Batas interpretasi")

    minimum_tick = int(data["elapsed_ticks"].min())
    st.write(
        f"Pengukuran tersingkat pada file ini adalah "
        f"{minimum_tick} tick. Durasi yang hanya beberapa tick "
        "memiliki keterbatasan ketelitian relatif."
    )

    st.write(
        "Perubahan jumlah thread antarpengukuran dapat memengaruhi "
        "overhead pengelolaan tim thread. Lima pengulangan "
        "menggambarkan variasi pada sesi ini dan belum membuktikan "
        "penyebab perubahan performa."
    )

    st.write(
        "Efficiency adalah rasio speedup terhadap jumlah worker. "
        "Angka ini bukan persentase penggunaan CPU."
    )

    st.write(
        "Konfigurasi tercepat berlaku untuk data, implementasi, "
        "perangkat, dan kondisi pengujian yang digunakan."
    )

if "log_terakhir" in st.session_state:
    with st.expander("Log pengujian C terakhir dari sesi web"):
        st.code(st.session_state["log_terakhir"], language="text")

st.divider()
st.caption(
    "Komputasi: C/OpenMP · Pengukuran: QPC · "
    "Pengolahan hasil: Pandas · Visualisasi: Plotly"
)
