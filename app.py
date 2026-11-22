"""
Dashboard Evaluasi Manifold Learning - Kemiripan Genetik Populasi Manusia

Aplikasi Streamlit satu halaman dengan tiga tab yang mengikuti urutan analisis:
1. Preprocessing          -> pipeline/preprocessing.py (menghasilkan feature matrix)
2. Distance Matrix        -> pipeline/distance.py
3. Embedding dan Evaluasi -> pipeline/reduction.py (UMAP, t-SNE, MDS, dan metrik)

File ini hanya lapisan presentasi. Komputasi berat dikerjakan modul pada folder
pipeline/ (bcftools, plink2, scikit-allel, umap-learn, scikit-learn) dan
hasilnya di-cache ke disk supaya tidak dihitung ulang setiap kali halaman
dimuat ulang. Dashboard dijalankan lokal oleh masing-masing pengguna:

    streamlit run app.py

Tata letak: Tab 1 dan Tab 2 memakai dua kolom sama lebar dengan tombol aksi di
bawahnya; Tab 3 memakai tiga kolom (filter dan kontrol, visualisasi, informasi
individu).
"""
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib import colormaps

from pipeline.preprocessing import (
    run_full_preprocessing, PreprocessingError, MAF_DEFAULT,
)
from pipeline.distance import compute_distance_matrix
from pipeline.reduction import run_reduction_and_evaluation, RANDOM_SEED_LOCKED

st.set_page_config(
    page_title="Evaluasi Manifold Learning - Kemiripan Genetik Populasi",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",  # sidebar tidak dipakai
)

SAMPLE_VCF_PATH = "sample_data/chr22_sample.vcf.gz"
SAMPLE_METADATA_PATH = "sample_data/metadata_1000genomes.csv"
CACHE_DIR = str(Path(__file__).resolve().parent / "cache")  # folder cache di samping app.py
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Step LD-pruning dikunci ke 1 (pilihan desain): window bergeser satu varian setiap kali.
LD_STEP_LOCKED = 1

# ============================================================
# TOKEN DESAIN
# Warna antarmuka (navbar, kartu statistik, latar) sengaja dipisah dari
# COLOR_MAP populasi di bawah, supaya warna pada grafik selalu bermakna
# "data", bukan "tampilan".
# ============================================================
NAVY = "#1E1B4B"
NAVY_LIGHT = "#302B7A"
PAGE_BG = "#F1F5F9"


def _load_logo_base64() -> str:
    """Baca sample_data/logo.png dari disk & encode base64 untuk header HTML."""
    logo_path = Path(__file__).parent / "sample_data" / "logo.png"
    if not logo_path.exists():
        return ""
    return base64.b64encode(logo_path.read_bytes()).decode("utf-8")


LOGO_B64 = _load_logo_base64()

# Batas ukuran file yang boleh dikirim ke browser lewat tombol unduh. File di
# atas batas ini tetap ada di folder cache/ (lihat pesan di UI).
DOWNLOAD_MAX_BYTES = 300 * 1024 * 1024


@st.cache_data(show_spinner=False)
def _read_file_bytes(path: str, mtime: float, size: int) -> bytes:
    """Baca file cache sekali saja. Streamlit menjalankan ulang seluruh script
    di setiap interaksi; tanpa cache ini, isi file akan diserialisasi ulang
    terus (lambat & boros memori untuk matriks besar). mtime & size ikut jadi
    kunci cache supaya hasil baru otomatis terbaca."""
    return Path(path).read_bytes()


def download_cached_file(label: str, path, file_name: str):
    """Tombol unduh dari file cache yang SUDAH ada di disk (tanpa serialisasi
    ulang). Kalau file terlalu besar untuk browser, tampilkan lokasinya."""
    if not path or not Path(path).exists():
        st.caption("File hasil belum tersedia di cache untuk diunduh.")
        return
    p = Path(path)
    stat = p.stat()
    if stat.st_size > DOWNLOAD_MAX_BYTES:
        st.caption(
            f"File terlalu besar untuk diunduh lewat browser "
            f"({stat.st_size / 1e6:,.0f} MB). Ambil langsung dari: {p.resolve()}"
        )
        return
    st.download_button(
        label, data=_read_file_bytes(str(p), stat.st_mtime, stat.st_size),
        file_name=file_name, use_container_width=True,
    )


def _fmt_table_value(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_striped_table(df: pd.DataFrame):
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows_html = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_fmt_table_value(val)}</td>" for val in row.tolist())
        rows_html.append(f"<tr>{cells}</tr>")
    table_html = (
        "<div class=\"striped-table-wrap\"><table class=\"striped-table\">"
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}

    .block-container {{ max-width: 1720px; padding-top: 1.05rem;
                        padding-left: 2.5rem; padding-right: 2.5rem; }}
    [data-testid="stAppViewContainer"], .stApp {{ background: {PAGE_BG}; }}

    /* Hilangkan elemen bawaan Streamlit pada tampilan final */
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
    #MainMenu, .stDeployButton, [data-testid="stDeployButton"], [data-testid="stStatusWidget"] {{ display:none !important; }}
    header {{ visibility:hidden; }}

    /* -- trik full-bleed: keluar dari padding block-container -- */
    .full-bleed {{
        margin-left: calc(-50vw + 50%); margin-right: calc(-50vw + 50%);
        width: auto; margin-bottom: 0;
    }}

    /* -- navbar: logo + judul, navy solid (bukan gradient - kesan formal) -- */
    .navbar-top {{ background: {NAVY}; padding: 1.05rem 2.85rem 1.05rem 2.85rem; min-height: 132px; display:flex; align-items:center; }}
    .navbar-brand {{ display: flex; align-items: center; gap: 1.15rem; padding-bottom: 0; width:100%; }}
    .navbar-logo {{ height: 84px; width: 72px; object-fit: contain; border-radius: 14px; background: #FFFFFF; padding: 5px; flex-shrink:0; }}
    .navbar-title {{ color: #F8FAFC; font-weight: 700; font-size: 1.48rem;
                     letter-spacing: -0.02em; line-height: 1.18; }}
    .navbar-subtitle {{ color: #DBEAFE; font-size: 0.9rem; line-height:1.45; }}

    /* -- strip tab: lebih lapang, highlight aktif lebih jelas. -- */
    div[data-baseweb="tab-list"] {{
        background: #FFFFFF; margin-top: 0.1rem; padding: 0.65rem 2.5rem 0.9rem 2.5rem;
        margin-left: calc(-50vw + 50%); margin-right: calc(-50vw + 50%); width: auto;
        gap: 0.75rem; border-bottom: 1px solid #D7DEE8; margin-bottom: 1.35rem;
    }}
    button[data-baseweb="tab"] {{
        font-weight: 600; font-size: 1rem; color: #475569;
        padding: 0.9rem 1.45rem 1rem 1.45rem;
        border-radius: 10px 10px 0 0;
        background: transparent;
        border-bottom: 3px solid transparent;
    }}
    button[data-baseweb="tab"]:hover {{ color: {NAVY}; background: #F8FAFC; }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {NAVY}; font-weight: 700; background: #E8EAFB;
        border-bottom: 4px solid #1E1B4B;
    }}
    div[data-baseweb="tab-highlight"] {{ background-color: transparent !important; height: 0 !important; }}

    /* -- kartu statistik: putih + aksen garis kiri, bukan blok gradient penuh --
          (blok gradient warna-warni terkesan seperti aplikasi konsumen; versi
          ini lebih sesuai nada dashboard akademik/analitik) -- */
    .stat-card-grid {{ display: grid; grid-template-columns: repeat(3, 1fr);
                       gap: 1rem; margin-bottom: 1.5rem; }}
    .stat-card {{ border-radius: 8px; padding: 0.95rem 1.2rem; background: #FFFFFF;
                 border: 1px solid #E2E8F0; }}
    .stat-card.c1 {{ border-left: 4px solid #2563EB; }}
    .stat-card.c2 {{ border-left: 4px solid #4338CA; }}
    .stat-card.c3 {{ border-left: 4px solid #475569; }}
    .stat-card .stat-label {{ font-size: 0.78rem; color: #64748B; margin-bottom: 0.35rem; }}
    .stat-card .stat-value {{ font-size: 1.5rem; font-weight: 700; color: #0F172A;
                              font-variant-numeric: tabular-nums; }}

    /* -- responsive fallback untuk layar sempit -- */
    @media (max-width: 1200px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .stat-card-grid {{ grid-template-columns: 1fr; }}
        .navbar-top {{ padding: 1rem 1.25rem 0 1.25rem; }}
        div[data-baseweb="tab-list"] {{ padding: 0 1.25rem; gap: 1rem; }}
    }}

    /* -- section label: sentence case, bukan ALL CAPS -- */
    .section-label {{
        font-size: 0.86rem; font-weight: 600; color: #1E293B;
        margin-bottom: 0.7rem; padding-bottom: 0.5rem; border-bottom: 1px solid #E2E8F0;
    }}
    .field-caption {{ font-size: 0.75rem; color: #94A3B8; margin-top: -0.6rem; margin-bottom: 0.8rem; }}

    /* -- panel: card putih melayang. -- */
    div[data-testid="stColumn"] div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 12px !important; border: 1px solid #E2E8F0 !important;
        background: #FFFFFF !important; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
        padding: 1rem 1.3rem !important;
    }}

    /* -- card per-chart di dalam grid visualisasi diberi tint abu sangat muda
          supaya terlihat jelas terpisah dari card putih induknya -- */
    .chart-card-wrap div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #F8FAFC !important; box-shadow: none !important;
    }}

    /* -- tombol primary: disamakan dengan header navy. -- */
    button[kind="primary"] {{
        background: {NAVY} !important; color: #FFFFFF !important; border: 1px solid {NAVY} !important;
        box-shadow: none !important;
    }}
    button[kind="primary"]:hover {{ background: #17153B !important; border-color: #17153B !important; }}
    button[kind="primary"]:disabled,
    button[kind="primary"][disabled],
    .stButton button:disabled,
    .stButton button[disabled] {{
        background: #CBD5E1 !important; color: #334155 !important; opacity: 1 !important;
        border: 1px solid #CBD5E1 !important; cursor: not-allowed !important;
    }}
    /* -- tombol download: bg lembut indigo -- */
    .stDownloadButton button {{ background: #EEF2FF !important; color: #3730A3 !important;
                                border: 1px solid #C7D2FE !important; }}

    /* -- kotak input nonaktif: border putus-putus, teks abu gelap
          supaya kontras terhadap background #F8FAFC memadai -- */
    div[data-testid="stNumberInput"] input:disabled {{
        border-style: dashed !important; background: #F8FAFC !important;
        color: #64748B !important; -webkit-text-fill-color: #64748B !important;
    }}

    /* -- segmented control 1D/2D/3D: radio horizontal dibuat mirip toggle kotak -- */
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        border: 1px solid #CBD5E1; border-radius: 6px; padding: 3px; background: #F8FAFC;
        display: inline-flex; gap: 0.25rem;
    }}
    div[data-testid="stRadio"] label {{ margin: 0 !important; }}

    /* -- header kecil per-chart -- */
    .chart-header {{ font-size: 0.85rem; font-weight: 600; color: #334155;
                     margin-bottom: 0.3rem; }}

    /* -- field kotak read-only di panel Informasi Individu -- */
    .info-field-label {{ font-size: 0.82rem; color: #64748B; margin-top: 0.9rem; margin-bottom: 0.2rem; }}
    .info-field-value {{ font-size: 0.95rem; color: #1E293B; font-weight: 500;
                         padding-bottom: 0.4rem; border-bottom: 1px solid #F1F5F9; min-height: 1.3rem;
                         overflow-wrap: break-word; word-break: break-word; }}

    /* -- tombol "Browse..." path VCF: diberi margin-top supaya sejajar
          secara visual dengan input teks di sebelahnya (bukan sejajar label) -- */
    .browse-btn-spacer {{ height: 1.7rem; }}

    /* -- kotak/card yang sejajar dalam satu baris kolom otomatis disamakan
          tingginya (dipakai di Tab 1: Input dataset & Parameter; Tab 2:
          Sumber data & Ringkasan hasil; Tab 3: Filter & kontrol,
          Visualisasi, Informasi individu). Flexbox stretch murni - tidak
          perlu height tetap, jadi tetap menyesuaikan konten terpanjang. -- */
    /* -- kotak/card yang sejajar dalam satu baris kolom otomatis disamakan
          tingginya (Tab 1: Input dataset & Parameter; Tab 2: Sumber data &
          Ringkasan hasil; Tab 3: Filter & kontrol, Visualisasi, Informasi
          individu). Dipakai rantai flexbox (flex:1), BUKAN height:100%,
          karena Streamlit menyisipkan div pembungkus ("element-container")
          di antara kolom & kartu yang defaultnya height:auto - memutus
          rantai persentase tinggi. flex:1 tidak bergantung pada tinggi
          pasti di tiap level, jadi rantainya tidak putus. -- */
    /* -- kotak/card yang sejajar dalam satu baris kolom otomatis disamakan
          tingginya (Tab 1: Input dataset & Parameter; Tab 2: Sumber data &
          Ringkasan hasil; Tab 3: Filter & kontrol, Visualisasi, Informasi
          individu). Rantai testid nyata (dicek lewat DevTools): stColumn >
          stVerticalBlock > stLayoutWrapper > (stElementContainer >)
          stVerticalBlockBorderWrapper. Dipakai flex:1 di tiap level (bukan
          height:100%) dan child-combinator (">") presisi mengikuti rantai
          itu persis, supaya widget lain yang lebih dalam (multiselect,
          expander, dst.) yang juga dibungkus stLayoutWrapper tidak ikut
          kena & jadi rusak tata letaknya. -- */
    div[data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
    div[data-testid="stColumn"] {{ display: flex; }}
    div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {{
        display: flex; flex-direction: column; flex: 1 1 auto; width: 100%;
    }}
    div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"]
        > div[data-testid="stLayoutWrapper"] {{
        display: flex; flex-direction: column; flex: 1 1 auto;
    }}
    div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"]
        > div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"]
        > div[data-testid="stLayoutWrapper"] > div[data-testid="stElementContainer"]
        > div[data-testid="stVerticalBlockBorderWrapper"] {{
        display: flex; flex-direction: column; flex: 1 1 auto;
    }}

    /* -- tabel ringkasan: selang-seling abu-putih agar lebih mudah dibaca -- */
    .striped-table-wrap {{
        width: 100%; overflow-x: auto; border: 1px solid #E2E8F0;
        border-radius: 10px; margin: 0.15rem 0 0.9rem 0; background: #FFFFFF;
    }}
    .striped-table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
    .striped-table thead th {{
        text-align: left; padding: 0.35rem 0.65rem; background: #F1F5F9;
        color: #334155; font-weight: 600; border-bottom: 1px solid #E2E8F0;
    }}
    .striped-table tbody td {{
        padding: 0.28rem 0.65rem; color: #1E293B; border-bottom: 1px solid #E2E8F0; height:28px;
        vertical-align: top;
    }}
    .striped-table tbody tr:nth-child(odd) td {{ background: #FFFFFF; }}
    .striped-table tbody tr:nth-child(even) td {{ background: #F8FAFC; }}
    .striped-table tbody tr:last-child td {{ border-bottom: none; }}
</style>
""", unsafe_allow_html=True)

# Warna tetap per populasi agar konsisten di setiap render.
COLOR_MAP = {
    "AFR": "#D85A30", "AMR": "#D4537E", "EAS": "#1D9E75",
    "EUR": "#7F77DD", "SAS": "#BA7517", "West_Java": "#EAB308",
}

# Bentuk marker per populasi sebagai penanda kedua selain warna (tetap terbaca
# pada cetak hitam-putih dan bagi pengguna buta warna). Plotly scatter_3d hanya
# menerima subset simbol (circle, circle-open, cross, diamond, diamond-open,
# square, square-open, x), sedangkan 1D/2D menerima simbol penuh, sehingga
# disediakan dua mapping: SYMBOL_MAP (1D/2D) dan SYMBOL_MAP_3D (3D).
SYMBOL_MAP = {
    "AFR": "circle", "AMR": "square", "EAS": "diamond",
    "EUR": "cross", "SAS": "triangle-up", "West_Java": "star",
}
FALLBACK_SYMBOLS = ["circle", "square", "diamond", "cross", "triangle-up",
                     "star", "x", "triangle-down", "pentagon", "hexagon"]

SYMBOL_MAP_3D = {
    "AFR": "circle", "AMR": "square", "EAS": "diamond",
    "EUR": "cross", "SAS": "x", "West_Java": "circle-open",
}
FALLBACK_SYMBOLS_3D = ["circle", "square", "diamond", "cross", "x",
                        "circle-open", "square-open", "diamond-open"]

FALLBACK_CMAP = colormaps.get_cmap("tab20")


def get_color(label: str, unknown_index: int) -> str:
    if label in COLOR_MAP:
        return COLOR_MAP[label]
    rgba = FALLBACK_CMAP(unknown_index % 20)
    return f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"


def get_symbol(label: str, unknown_index: int, is_3d: bool = False) -> str:
    mapping = SYMBOL_MAP_3D if is_3d else SYMBOL_MAP
    fallback = FALLBACK_SYMBOLS_3D if is_3d else FALLBACK_SYMBOLS
    if label in mapping:
        return mapping[label]
    return fallback[unknown_index % len(fallback)]


def _read_simple_metadata_csv(file_or_path) -> pd.DataFrame | None:
    """Parse CSV metadata sederhana: wajib kolom sample_id, population."""
    df = pd.read_csv(file_or_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"sample_id", "population"}.issubset(df.columns):
        return None
    df["sample_id"] = df["sample_id"].astype(str)
    return df[["sample_id", "population"]]


def _browse_for_vcf_file() -> str:
    """Buka dialog pemilihan file native OS via tkinter.

    HANYA berfungsi kalau proses Streamlit ini berjalan di mesin yang sama
    dengan tampilan desktop yang dipakai user (skenario `streamlit run
    app.py` lokal biasa). Browser TIDAK PERNAH mengizinkan web app membaca
    path filesystem asli dari <input type="file"> (sandboxing keamanan) -
    ini kenapa trik ini lewat proses server (tkinter), bukan lewat elemen
    HTML biasa. Kalau tidak ada display (mis. server headless, WSL tanpa
    X server, atau deployment jarak jauh), fungsi ini akan melempar
    exception dan pemanggil harus menampilkan pesan fallback yang jelas.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Pilih file VCF",
        filetypes=[("VCF (bgzip)", "*.vcf.gz"), ("VCF", "*.vcf"), ("Semua file", "*.*")],
    )
    root.destroy()
    return path


def init_state():
    defaults = {
        "feature_matrix": None,
        "sample_ids": None,
        "sample_ids_signature": None,   # dipakai deteksi pergantian dataset
        "feature_matrix_cache_path": None,
        "preprocessing_summary": None,
        "distance_matrix": None,
        "distance_cache_path": None,
        "_last_dist_result": None,
        "metadata_raw_df": None,       # metadata mentah (belum dicocokkan ID)
        "metadata_source_label": None,  # nama sumber, buat ditampilkan ke user
        "metadata_df": None,            # metadata yang SUDAH dicocokkan ke sample_ids
        "reduction_results": None,
        "selected_sample": None,
        "vcf_path_input": "",
        "vcf_browse_unavailable": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _sample_ids_signature(ids):
    return tuple(str(i) for i in ids) if ids is not None else None


def sync_sample_ids(new_ids):
    """Panggil setiap kali sample_ids baru tersedia (dari preprocessing baru
    maupun upload manual). Kalau dataset benar-benar berganti (urutan/isi
    sample_ids berbeda dari sebelumnya), semua state turunan yang bergantung
    pada pencocokan/urutan sampel LAMA di-reset - supaya tidak ada label
    populasi atau titik terpilih yang "nyangkut" dan diam-diam salah
    dipasangkan ke individu yang berbeda.
    """
    new_sig = _sample_ids_signature(new_ids)
    old_sig = st.session_state.get("sample_ids_signature")
    if new_ids is not None and new_sig != old_sig:
        st.session_state.metadata_df = None
        st.session_state.selected_sample = None
        st.session_state.reduction_results = None
        st.session_state.sample_ids_signature = new_sig
    st.session_state.sample_ids = new_ids


init_state()

# ============================================================
# NAVBAR
# ============================================================
_logo_img_tag = (
    f'<img class="navbar-logo" src="data:image/png;base64,{LOGO_B64}" alt="Logo Universitas Riau">'
    if LOGO_B64 else ""
)
st.markdown(
    f'<div class="full-bleed navbar-top"><div class="navbar-brand">'
    f'{_logo_img_tag}'
    f'<div><div class="navbar-title">Dashboard Evaluasi Algoritma Manifold Learning</div>'
    f'<div class="navbar-subtitle">Visualisasi Kemiripan Genetik Populasi Manusia<br><span style="font-size:0.82rem;color:#DBEAFE;">Universitas Riau</span></div>'
    f'</div></div></div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Preprocessing", "Distance Matrix", "Embedding dan Evaluasi"])

# ============================================================
# TAB 1 - PREPROCESSING
# Alur: [Input dataset] [Parameter] -> tombol di tengah -> ringkasan hasil
# (ringkasan HANYA dirender setelah ada hasil, tidak ada kotak kosong).
# ============================================================
with tab1:
    col_input, col_param = st.columns([1, 1])

    with col_input:
        with st.container(border=True):
            st.markdown('<div class="section-label">Input dataset</div>', unsafe_allow_html=True)

            def _on_toggle_sample():
                if st.session_state.get("use_sample_checkbox"):
                    st.session_state["vcf_path_input"] = SAMPLE_VCF_PATH
                elif st.session_state.get("vcf_path_input") == SAMPLE_VCF_PATH:
                    st.session_state["vcf_path_input"] = ""

            use_sample = st.checkbox(
                "Gunakan data contoh",
                key="use_sample_checkbox",
                on_change=_on_toggle_sample,
                help="Menggunakan subset kromosom 22 dari 1000 Genomes untuk demo cepat, beserta metadata populasinya.",
            )

            path_col, browse_col = st.columns([5, 1])
            with path_col:
                vcf_path = st.text_input(
                    "Path file VCF (.vcf / .vcf.gz)",
                    key="vcf_path_input",
                    disabled=use_sample,
                    help="Contoh: /home/user/data/chr22.vcf.gz atau chr22.vcf",
                )
            with browse_col:
                st.markdown('<div class="browse-btn-spacer"></div>', unsafe_allow_html=True)
                if st.button("📁 Browse", disabled=use_sample, use_container_width=True,
                              help="Buka dialog pemilihan file OS. Hanya berfungsi kalau dashboard "
                                   "dijalankan lokal dengan tampilan desktop (server & browser di "
                                   "mesin yang sama)."):
                    try:
                        selected_path = _browse_for_vcf_file()
                        st.session_state["vcf_browse_unavailable"] = False
                        if selected_path:
                            st.session_state["vcf_path_input"] = selected_path
                            st.rerun()
                    except Exception:
                        st.session_state["vcf_browse_unavailable"] = True

            if st.session_state.get("vcf_browse_unavailable"):
                st.info(
                    "Dialog pemilihan file tidak tersedia di lingkungan ini (butuh tampilan "
                    "desktop / X server - misalnya tidak tersedia begitu saja saat dijalankan "
                    "lewat WSL tanpa X server dikonfigurasi). Ketik path secara manual, atau "
                    "salin path-nya dari File Explorer/Finder."
                )

            st.markdown(
                '<div class="field-caption">Format: .vcf atau .vcf.gz (bgzip; index dibuat otomatis), '
                'build hg19/GRCh37. Filter biallelic SNP dan filter missing genotype dilakukan otomatis.</div>',
                unsafe_allow_html=True,
            )

            st.markdown("Metadata populasi (CSV)")
            meta_upload_1 = st.file_uploader(
                "Unggah file .csv", type=["csv"], disabled=use_sample,
                label_visibility="collapsed",
                help="Kolom wajib: sample_id, population.",
            )

            if use_sample:
                sample_meta = _read_simple_metadata_csv(SAMPLE_METADATA_PATH)
                st.session_state.metadata_raw_df = sample_meta
                st.session_state.metadata_source_label = "metadata_1000genomes.csv (data contoh)"
            elif meta_upload_1 is not None:
                parsed = _read_simple_metadata_csv(meta_upload_1)
                if parsed is None:
                    st.error("CSV metadata harus punya kolom 'sample_id' dan 'population'.")
                else:
                    st.session_state.metadata_raw_df = parsed
                    st.session_state.metadata_source_label = meta_upload_1.name

    with col_param:
        with st.container(border=True):
            st.markdown('<div class="section-label">Parameter preprocessing</div>', unsafe_allow_html=True)

            maf_threshold = st.number_input(
                "Ambang MAF",
                min_value=0.0,
                max_value=0.5,
                value=MAF_DEFAULT,
                step=0.01,
                format="%.2f",
                help="Default 0,05 digunakan pada eksperimen utama. Nilai ini dapat diubah sesuai kebutuhan analisis.",
            )
            pcol_a, pcol_b = st.columns(2)
            with pcol_a:
                r2_threshold = st.number_input(
                    "Ambang r² LD-pruning",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                )
            with pcol_b:
                window_size = st.number_input(
                    "Window size (jumlah SNP)",
                    min_value=50,
                    max_value=20000,
                    value=1000,
                    step=50,
                    help="Ukuran window LD-pruning dalam JUMLAH VARIAN (bukan kb atau jarak fisik).",
                )

            pcol_c, pcol_d = st.columns(2)
            with pcol_c:
                st.number_input(
                    "Step size LD-pruning (terkunci)",
                    value=LD_STEP_LOCKED,
                    disabled=True,
                    help="Dikunci ke 1 supaya window bergeser satu varian setiap kali.",
                )
            with pcol_d:
                max_snps_input = st.number_input(
                    "Batas maksimum SNP (0 = tanpa batas)",
                    min_value=0, value=0, step=1000,
                    help="Kalau jumlah SNP setelah LD-pruning melebihi batas ini, yang dipertahankan "
                         "adalah SNP dengan MAF tertinggi (posisi aslinya tetap berurutan). Berguna "
                         "untuk whole genome supaya feature matrix muat di RAM.",
                )

    st.write("")
    _, btn_col1, _ = st.columns([1, 1.2, 1])
    with btn_col1:
        prep_disabled = (not use_sample) and (not vcf_path or not vcf_path.strip())
        run_prep = st.button(
            "Jalankan preprocessing", type="primary", use_container_width=True,
            disabled=prep_disabled,
        )
        if prep_disabled:
            st.caption("Isi path file VCF terlebih dahulu, atau centang \"Gunakan data contoh\".")

    prep_status_area = st.empty()

    if run_prep:
        try:
            with st.spinner("Memproses... (bisa lama untuk whole genome)"):
                def report(msg):
                    prep_status_area.info(msg)

                result = run_full_preprocessing(
                    vcf_path=vcf_path,
                    workdir=CACHE_DIR,
                    r2=r2_threshold,
                    window_variants=int(window_size),
                    step=LD_STEP_LOCKED,
                    maf=float(maf_threshold),
                    max_snps=int(max_snps_input) or None,
                    progress_callback=report,
                )
            prep_status_area.empty()

            # dataset (feature matrix) berganti -> distance matrix & hasil
            # reduksi lama sudah tidak relevan, metadata perlu dicocokkan ulang.
            sync_sample_ids(result["sample_ids"])
            st.session_state.feature_matrix = result["feature_matrix"]
            st.session_state.feature_matrix_cache_path = result["feature_matrix_cache_path"]
            st.session_state.preprocessing_summary = result
            st.session_state.distance_matrix = None
            st.session_state.distance_cache_path = None
            st.session_state["_last_dist_result"] = None
        except PreprocessingError as e:
            prep_status_area.error(str(e))
        except FileNotFoundError as e:
            prep_status_area.error(f"File tidak ditemukan: {e}")
        except Exception as e:  # jangan tampilkan traceback mentah ke pengguna
            prep_status_area.error("Terjadi error tak terduga saat preprocessing.")
            with prep_status_area.container():
                with st.expander("Detail teknis error (untuk debugging)"):
                    st.exception(e)

    # -- Ringkasan hasil: HANYA dirender kalau sudah ada hasil, supaya tidak
    #    ada kotak kosong yang mengganggu sebelum preprocessing dijalankan. --
    if st.session_state.preprocessing_summary:
        st.divider()
        with st.container(border=True):
            st.markdown('<div class="section-label">Ringkasan hasil</div>', unsafe_allow_html=True)

            r = st.session_state.preprocessing_summary

            n_samples_disp = f"{r['n_samples']:,}".replace(",", ".")
            n_snp_disp = f"{r['n_after_ld_prune']:,}".replace(",", ".")
            n_removed_disp = f"{r['n_removed_missing']:,}".replace(",", ".")
            st.markdown(f"""
            <div class="stat-card-grid">
                <div class="stat-card c1"><div class="stat-label">Total individu</div>
                    <div class="stat-value">{n_samples_disp}</div></div>
                <div class="stat-card c2"><div class="stat-label">Total SNP (pasca pruning)</div>
                    <div class="stat-value">{n_snp_disp}</div></div>
                <div class="stat-card c3"><div class="stat-label">SNP dibuang (missing)</div>
                    <div class="stat-value">{n_removed_disp}</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Tabel jumlah varian per tahap. Kolomnya "Jumlah varian" (bukan "SNP")
            # karena varian mentah masih memuat indel/multiallelic/SV.
            rows = [
                {"Tahap": "Varian mentah", "Jumlah varian": r["n_variants_raw"]},
                {"Tahap": "Setelah filter biallelic SNP", "Jumlah varian": r["n_after_biallelic_snp"]},
                {"Tahap": f"Setelah filter MAF ({r.get('maf_threshold', MAF_DEFAULT)*100:g}%)", "Jumlah varian": r["n_after_maf"]},
                {"Tahap": "Setelah filter missing genotype", "Jumlah varian": r["n_after_missing_filter"]},
                {"Tahap": "Setelah LD-pruning", "Jumlah varian": r["n_after_ld_prune"]},
            ]
            if r.get("max_snps") and r["n_after_cap"] < r["n_after_ld_prune"]:
                cap_txt = f"{r['max_snps']:,}".replace(",", ".")
                rows.append({
                    "Tahap": f"Setelah batas maks. SNP (MAF tertinggi, maks {cap_txt})",
                    "Jumlah varian": r["n_after_cap"],
                })
            summary_df = pd.DataFrame(rows)
            render_striped_table(summary_df)

            # File .npz sudah ditulis oleh pipeline (berisi feature_matrix &
            # sample_ids) - cukup dikirim, tidak diserialisasi ulang tiap rerun.
            download_cached_file(
                "Unduh feature matrix (.npz)",
                r["feature_matrix_cache_path"], "feature_matrix.npz",
            )
            st.success("Preprocessing selesai.")

# ============================================================
# TAB 2 - HITUNG JARAK
# Alur sama seperti Tab 1: [Sumber data] [Tentang metrik] -> tombol
# di tengah -> ringkasan hasil (baru muncul setelah dihitung).
# ============================================================
with tab2:
    # Kolom kanan ("Ringkasan hasil") selalu tampil; isinya placeholder sebelum
    # distance matrix dihitung.
    col_source, col_summary = st.columns([1, 1])

    feature_matrix = None
    sample_ids = None
    source_key_for_cache = None

    with col_source:
        with st.container(border=True):
            st.markdown('<div class="section-label">Sumber data</div>', unsafe_allow_html=True)

            source_choice = st.radio(
                "Sumber feature matrix",
                ["Pakai hasil Tab 1 (sesi ini)", "Upload feature matrix sendiri (.npz)"],
            )

            if source_choice == "Pakai hasil Tab 1 (sesi ini)":
                if st.session_state.feature_matrix is not None:
                    feature_matrix = st.session_state.feature_matrix
                    sample_ids = st.session_state.sample_ids
                    source_key_for_cache = st.session_state.feature_matrix_cache_path
                    st.info(
                        f"Feature matrix tersedia: {feature_matrix.shape[0]} individu "
                        f"× {feature_matrix.shape[1]} SNP."
                    )
                else:
                    st.warning("Belum ada hasil dari Tab 1 di sesi ini. Jalankan Tab 1 dulu, atau upload manual.")
            else:
                uploaded_npz = st.file_uploader("Upload feature_matrix.npz", type=["npz"])
                if uploaded_npz is not None:
                    data = np.load(uploaded_npz, allow_pickle=True)
                    feature_matrix = data["feature_matrix"]
                    sample_ids = data["sample_ids"].tolist() if "sample_ids" in data else None
                    tmp_path = f"{CACHE_DIR}/uploaded_feature_matrix.npz"
                    with open(tmp_path, "wb") as f:
                        uploaded_npz.seek(0)
                        f.write(uploaded_npz.read())
                    source_key_for_cache = tmp_path
                    st.info(
                        f"Feature matrix ter-upload: {feature_matrix.shape[0]} individu "
                        f"× {feature_matrix.shape[1]} SNP."
                    )


    with col_summary:
        with st.container(border=True):
            st.markdown('<div class="section-label">Ringkasan hasil</div>', unsafe_allow_html=True)

            if st.session_state.distance_matrix is not None:
                dr = st.session_state.get("_last_dist_result")
                shape = dr["shape"] if dr else st.session_state.distance_matrix.shape
                is_sym = dr["is_symmetric"] if dr else None
                diag_zero = dr["diagonal_is_zero"] if dr else None
                summary_df = pd.DataFrame([
                    {"Properti": "Ukuran matriks", "Nilai": f"{shape[0]} × {shape[1]}"},
                    {"Properti": "Simetris?", "Nilai": "Ya" if is_sym else ("Tidak" if is_sym is not None else "—")},
                    {"Properti": "Diagonal nol?", "Nilai": "Ya" if diag_zero else ("Tidak" if diag_zero is not None else "—")},
                ])
                render_striped_table(summary_df)

                download_cached_file(
                    "Unduh distance matrix (.npy)",
                    st.session_state.distance_cache_path, "distance_matrix.npy",
                )
                st.success("Distance matrix selesai dihitung.")
            else:
                st.caption(
                    "Area pesan validasi & ringkasan hasil akan muncul di sini "
                    "setelah distance matrix dihitung."
                )

    st.write("")
    _, btn_col2, _ = st.columns([1, 1.2, 1])
    with btn_col2:
        run_dist = st.button(
            "Hitung distance matrix", type="primary",
            use_container_width=True, disabled=feature_matrix is None,
        )
        if feature_matrix is None:
            st.caption("Pilih atau siapkan feature matrix terlebih dahulu.")

    if run_dist and feature_matrix is not None:
        try:
            with st.spinner("Menghitung distance matrix Euclidean..."):
                dist_result = compute_distance_matrix(
                    feature_matrix, CACHE_DIR, source_key=source_key_for_cache
                )
            # simpan feature_matrix yang benar-benar dipakai (termasuk jalur
            # upload manual) supaya Tab 3 bisa menampilkan jumlah SNP akurat.
            st.session_state.feature_matrix = feature_matrix
            # kalau sample_ids berganti (dataset baru), reset otomatis semua
            # state turunan yang bergantung pada pencocokan/urutan sampel lama.
            if sample_ids is not None:
                sync_sample_ids(sample_ids)
            else:
                st.session_state.metadata_df = None
                st.session_state.selected_sample = None
                st.session_state.reduction_results = None
            st.session_state.distance_matrix = dist_result["distance_matrix"]
            st.session_state.distance_cache_path = dist_result["cache_path"]
            st.session_state["_last_dist_result"] = dist_result
            # Ringkasan hasil dirender di kolom kanan, di atas blok ini. Tanpa
            # st.rerun(), isinya baru ter-update pada interaksi berikutnya.
            st.rerun()
        except Exception as e:  # tangkap error tak terduga (mis. file .npz rusak)
            st.error("Terjadi error tak terduga saat menghitung distance matrix.")
            with st.expander("Detail teknis error (untuk debugging)"):
                st.exception(e)

# ============================================================
# TAB 3 - REDUKSI DIMENSI & VISUALISASI
# ============================================================
with tab3:
    if st.session_state.distance_matrix is None:
        st.warning("Belum ada distance matrix. Jalankan Tab 2 dulu.")
    else:
        n_samples = st.session_state.distance_matrix.shape[0]
        sample_ids = st.session_state.sample_ids or [f"sample_{i}" for i in range(n_samples)]

        # --- metadata: pakai yang sudah diisi dari Tab 1 kalau ada, kalau tidak
        #     tawarkan upload di sini juga (supaya tab tetap bisa dilangkau) ---
        if st.session_state.metadata_raw_df is not None and st.session_state.metadata_df is None:
            lookup = dict(zip(
                st.session_state.metadata_raw_df["sample_id"],
                st.session_state.metadata_raw_df["population"],
            ))
            missing_ids = [sid for sid in sample_ids if str(sid) not in lookup]
            if missing_ids:
                st.error(
                    f"Metadata dari '{st.session_state.metadata_source_label}': "
                    f"{len(missing_ids)} sampel tidak ditemukan. Contoh: {missing_ids[:5]}"
                )
            else:
                st.session_state.metadata_df = pd.DataFrame({
                    "sample_id": [str(s) for s in sample_ids],
                    "population": [lookup[str(s)] for s in sample_ids],
                })

        with st.expander(
            f"Metadata populasi — {st.session_state.metadata_source_label or 'belum ada'}",
            expanded=st.session_state.metadata_df is None,
        ):
            if st.session_state.metadata_df is not None:
                st.success(
                    f"Metadata aktif: {st.session_state.metadata_source_label} "
                    f"({len(st.session_state.metadata_df)} sampel cocok). "
                    "Untuk ganti, upload ulang di Tab 1, atau upload file lain di bawah."
                )
            else:
                st.warning(
                    "Belum ada metadata populasi. Tanpa label ground-truth, "
                    "metrik Silhouette Score dan Davies-Bouldin Index tidak dapat "
                    "dihitung / tidak akan valid. Trustworthiness tetap bisa dihitung."
                )
            st.caption(
                "Format resmi 1000 Genomes (.panel, kolom sample/pop/super_pop) juga didukung."
            )
            meta_file = st.file_uploader(
                "Upload metadata populasi (.csv, atau .panel gaya 1000 Genomes)",
                type=["csv", "panel", "txt"], key="meta_upload_tab3",
            )
            if meta_file is not None:
                filename = meta_file.name.lower()
                is_panel = filename.endswith(".panel") or filename.endswith(".txt")
                if is_panel:
                    raw_df = pd.read_csv(meta_file, sep="\t")
                    raw_df.columns = [c.strip().lower() for c in raw_df.columns]
                    if "sample" not in raw_df.columns:
                        st.error(f"File .panel harus punya kolom 'sample' (ditemukan: {list(raw_df.columns)}).")
                        meta_df_new = None
                    else:
                        pop_col_options = [c for c in ["super_pop", "pop"] if c in raw_df.columns]
                        if not pop_col_options:
                            st.error("File .panel harus punya kolom 'pop' atau 'super_pop'.")
                            meta_df_new = None
                        else:
                            chosen_col = st.radio(
                                "Level populasi yang dipakai sebagai label",
                                pop_col_options, horizontal=True,
                                help="'super_pop' = 5 grup besar, 'pop' = sub-populasi lebih rinci.",
                            )
                            meta_df_new = pd.DataFrame({
                                "sample_id": raw_df["sample"].astype(str),
                                "population": raw_df[chosen_col],
                            })
                else:
                    meta_df_new = _read_simple_metadata_csv(meta_file)
                    if meta_df_new is None:
                        st.error("CSV metadata harus punya kolom 'sample_id' dan 'population'.")

                if meta_df_new is not None:
                    lookup = dict(zip(meta_df_new["sample_id"].astype(str), meta_df_new["population"]))
                    missing_ids = [sid for sid in sample_ids if str(sid) not in lookup]
                    if missing_ids:
                        st.error(f"{len(missing_ids)} sampel tidak ditemukan di metadata. Contoh: {missing_ids[:5]}")
                    else:
                        st.session_state.metadata_df = pd.DataFrame({
                            "sample_id": [str(s) for s in sample_ids],
                            "population": [lookup[str(s)] for s in sample_ids],
                        })
                        st.session_state.metadata_source_label = meta_file.name
                        st.success("Metadata cocok dengan seluruh sampel.")

        labels = None
        if st.session_state.metadata_df is not None:
            labels = st.session_state.metadata_df["population"].tolist()
            unique_pops = sorted(set(labels))
        else:
            unique_pops = []

        active_fm = st.session_state.feature_matrix
        n_snp_display = (
            f"{active_fm.shape[1]:,}".replace(",", ".")
            if active_fm is not None and active_fm.shape[0] == n_samples
            else "—"
        )
        n_samples_display = f"{n_samples:,}".replace(",", ".")
        st.markdown(f"""
        <div class="stat-card-grid">
            <div class="stat-card c1"><div class="stat-label">Total individu</div>
                <div class="stat-value">{n_samples_display}</div></div>
            <div class="stat-card c2"><div class="stat-label">Total SNP (pasca pruning)</div>
                <div class="stat-value">{n_snp_display}</div></div>
            <div class="stat-card c3"><div class="stat-label">Total populasi</div>
                <div class="stat-value">{len(unique_pops) if unique_pops else "—"}</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Kolom visualisasi dibuat paling lebar; kolom "Informasi individu"
        # sempit karena isinya hanya empat field. Ketiga kotak otomatis sama
        # tinggi lewat CSS stretch di atas.
        left_col, mid_col, right_col = st.columns([1, 3.8, 0.6])

        with left_col:
          with st.container(border=True):
            st.markdown('<div class="section-label">Filter & kontrol</div>', unsafe_allow_html=True)
            filter_pop = st.multiselect(
                "Filter populasi", options=unique_pops, default=unique_pops,
            ) if unique_pops else []
            search_id = st.text_input("Cari ID sampel", value="", placeholder="Ketik ID...")
            dimensi = st.radio("Dimensi", ["1D", "2D", "3D"], index=1, horizontal=True)
            n_components = {"1D": 1, "2D": 2, "3D": 3}[dimensi]

            st.markdown("Tampilkan metode")
            show_umap = st.checkbox("UMAP", value=True)
            show_tsne = st.checkbox("t-SNE", value=True)
            show_mds = st.checkbox("MDS", value=True)
            show_methods = [m for m, v in
                             [("UMAP", show_umap), ("t-SNE", show_tsne), ("MDS", show_mds)] if v]

            with st.expander("Parameter lanjutan"):
                umap_n_neighbors = st.slider("UMAP n_neighbors", 2, 100, 15)
                umap_min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1)
                max_perp = max(2, min(100, n_samples - 1))
                default_perp = min(30, max_perp)
                tsne_perplexity = st.slider(
                    "t-SNE perplexity", 2, max_perp, default_perp,
                    help="Perplexity harus lebih kecil dari jumlah sampel."
                )
                mds_n_init = st.slider(
                    "MDS n_init", 1, 10, 4,
                    help="Berapa kali MDS dijalankan dari posisi awal acak yang "
                         "berbeda; hasil dengan stress terkecil yang dipakai."
                )
                st.caption(f"Random seed dikunci: {RANDOM_SEED_LOCKED}")

            if n_samples < 30 and dimensi == "3D":
                st.warning("Jumlah sampel sedikit, visualisasi 3D mungkin kurang informatif.")

            viz_disabled = (
                len(show_methods) == 0
                or (show_tsne and tsne_perplexity >= n_samples)
            )
            run_viz = st.button(
                "Jalankan visualisasi", type="primary", use_container_width=True,
                disabled=viz_disabled,
            )
            if len(show_methods) == 0:
                st.caption("Pilih minimal satu metode untuk divisualisasikan.")
            elif show_tsne and tsne_perplexity >= n_samples:
                st.caption("Perplexity t-SNE harus lebih kecil dari jumlah sampel.")

            # Reset sesi: menghapus seluruh state (feature matrix, distance matrix, hasil).
            st.divider()
            if st.button("Reset sesi", use_container_width=True,
                          help="Menghapus semua data & hasil di sesi ini untuk mulai dari awal."):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
            st.caption(
                "Menghapus feature matrix, distance matrix, dan hasil visualisasi "
                "yang tersimpan sementara di sesi ini."
            )

        if run_viz:
            status_box = mid_col.empty()
            try:
                with st.spinner("Menjalankan reduksi dimensi..."):
                    def report(msg):
                        status_box.info(msg)

                    results = run_reduction_and_evaluation(
                        st.session_state.distance_matrix, CACHE_DIR,
                        source_key=st.session_state.distance_cache_path,
                        labels=labels, n_components=n_components,
                        umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist,
                        tsne_perplexity=tsne_perplexity,
                        mds_n_init=mds_n_init,
                        progress_callback=report,
                    )
                status_box.empty()
                st.session_state.reduction_results = results
                st.session_state.selected_sample = None  # hasil baru -> titik terpilih lama tidak relevan lagi
            except Exception as e:  # tangkap error tak terduga dari pipeline reduksi
                status_box.error("Terjadi error tak terduga saat menjalankan reduksi dimensi.")
                with mid_col:
                    with st.expander("Detail teknis error (untuk debugging)"):
                        st.exception(e)

        with mid_col:
          with st.container(border=True):
            st.markdown('<div class="section-label">Visualisasi reduksi dimensi</div>', unsafe_allow_html=True)
            results = st.session_state.reduction_results
            selected_methods_preview = [m for m in ["UMAP", "t-SNE", "MDS"] if m in show_methods]
            results_dim_matches = (
                results is not None
                and all(
                    results[m]["embedding"].shape[1] == n_components
                    for m in selected_methods_preview if m in results
                )
            )
            if results is None:
                st.caption("Klik \"Jalankan Visualisasi\" untuk menghasilkan scatter plot.")
            elif not results_dim_matches:
                st.info(
                    "Dimensi diubah sejak visualisasi terakhir dijalankan — "
                    "klik \"Jalankan Visualisasi\" lagi untuk menghitung ulang embedding."
                )
            else:
                # filter sampel sesuai populasi & pencarian ID
                mask = np.ones(n_samples, dtype=bool)
                if unique_pops and filter_pop:
                    label_arr = np.array(labels)
                    mask &= np.isin(label_arr, filter_pop)
                if search_id.strip():
                    id_arr = np.array([str(s) for s in sample_ids])
                    mask &= np.char.find(id_arr.astype(str), search_id.strip()) >= 0

                if mask.sum() == 0:
                    st.info("Tidak ada sampel yang cocok dengan filter populasi / pencarian ID saat ini.")
                else:
                    # Pilih mapping simbol sesuai dimensi (3D hanya menerima subset
                    # simbol; lihat SYMBOL_MAP_3D di atas).
                    is_3d_mode = n_components == 3
                    pop_to_color = {}
                    pop_to_symbol = {}
                    for i, p in enumerate(unique_pops):
                        pop_to_color[p] = get_color(p, i)
                        pop_to_symbol[p] = get_symbol(p, i, is_3d=is_3d_mode)

                    # PENTING: jangan set title=None di sini. Beberapa versi Plotly.js
                    # merender title None sebagai teks literal "undefined" di atas chart
                    # (bug Plotly.js, bukan salah logika) - defaultnya memang sudah tanpa
                    # judul (kita pakai header HTML custom di atas chart), jadi cukup
                    # tidak menyertakan key "title" sama sekali.
                    PLOT_LAYOUT = dict(
                        template="plotly_white",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=340,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                                    font=dict(size=10), itemsizing="constant"),
                        font=dict(family="sans-serif", size=11, color="#334155"),
                        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                    )
                    AXIS_STYLE = dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                                       linecolor="#E2E8F0")

                    selected_methods = [m for m in ["UMAP", "t-SNE", "MDS"] if m in show_methods]

                    chart_slots = []
                    for i in range(0, len(selected_methods), 2):
                        row_methods = selected_methods[i:i + 2]
                        if len(row_methods) == 2:
                            row_cols = st.columns(2)
                        else:
                            _spacer_l, center_col, _spacer_r = st.columns([1, 2, 1])
                            row_cols = [center_col]
                        chart_slots += list(zip(row_methods, row_cols))

                    for method, col_slot in chart_slots:
                        with col_slot:
                            st.markdown('<div class="chart-card-wrap">', unsafe_allow_html=True)
                            with st.container(border=True):
                                st.markdown(f'<div class="chart-header">{method}</div>', unsafe_allow_html=True)
                                emb = results[method]["embedding"][mask]
                                ids_f = np.array([str(s) for s in sample_ids])[mask]
                                pops_f = (np.array(labels)[mask] if labels is not None
                                          else np.array(["(tanpa label)"] * mask.sum()))

                                df_plot = pd.DataFrame({"sample_id": ids_f, "population": pops_f})
                                for d in range(n_components):
                                    df_plot[f"dim{d+1}"] = emb[:, d]

                                color_map_used = pop_to_color if unique_pops else None
                                symbol_map_used = pop_to_symbol if unique_pops else None

                                if n_components == 1:
                                    df_plot["y_jitter"] = np.random.default_rng(0).normal(0, 0.05, size=len(df_plot))
                                    fig = px.strip(
                                        df_plot, x="dim1", y="y_jitter", color="population",
                                        symbol="population" if symbol_map_used else None,
                                        color_discrete_map=color_map_used,
                                        symbol_map=symbol_map_used,
                                        hover_data=["sample_id"],
                                    )
                                elif n_components == 2:
                                    fig = px.scatter(
                                        df_plot, x="dim1", y="dim2", color="population",
                                        symbol="population" if symbol_map_used else None,
                                        color_discrete_map=color_map_used,
                                        symbol_map=symbol_map_used,
                                        hover_data=["sample_id"],
                                    )
                                else:
                                    fig = px.scatter_3d(
                                        df_plot, x="dim1", y="dim2", z="dim3", color="population",
                                        symbol="population" if symbol_map_used else None,
                                        color_discrete_map=color_map_used,
                                        symbol_map=symbol_map_used,
                                        hover_data=["sample_id"],
                                    )

                                fig.update_traces(marker=dict(size=7, opacity=0.85,
                                                               line=dict(width=0.5, color="white")))
                                fig.update_layout(**PLOT_LAYOUT)
                                fig.update_xaxes(**AXIS_STYLE)
                                fig.update_yaxes(**AXIS_STYLE)

                                # Klik-pilih titik pada scatter 3D Plotly belum stabil, jadi
                                # untuk 3D klik dinonaktifkan (detail individu lewat hover).
                                if n_components == 3:
                                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{method}")
                                    st.caption("Mode 3D: detail individu lewat hover, klik belum didukung.")
                                else:
                                    event = st.plotly_chart(
                                        fig, use_container_width=True, key=f"plot_{method}",
                                        on_select="rerun", selection_mode="points",
                                    )
                                    if event and event.get("selection", {}).get("points"):
                                        pt = event["selection"]["points"][0]
                                        pt_idx = pt.get("point_index")
                                        if pt_idx is not None and pt_idx < len(df_plot):
                                            row = df_plot.iloc[pt_idx]
                                            coords = [row[f"dim{d+1}"] for d in range(n_components)]
                                            st.session_state.selected_sample = {
                                                "sample_id": row["sample_id"],
                                                "population": row["population"],
                                                "method": method,
                                                "coords": coords,
                                            }
                            st.markdown('</div>', unsafe_allow_html=True)

                    st.write("")
                    st.markdown('<div class="section-label">Tabel metrik evaluasi</div>', unsafe_allow_html=True)
                    metric_rows = []
                    for method in selected_methods:
                        m = results[method]["metrics"]
                        metric_rows.append({
                            "Metode": method,
                            "Silhouette": m["silhouette"],
                            "Silhouette (macro-avg)": m["silhouette_macro"],
                            "Davies-Bouldin": m["davies_bouldin"],
                            "Trustworthiness": m["trustworthiness"],
                        })
                    metric_df = pd.DataFrame(metric_rows)

                    numeric_cols = ["Silhouette", "Silhouette (macro-avg)", "Davies-Bouldin", "Trustworthiness"]
                    metric_display_df = metric_df.copy()
                    metric_display_df[numeric_cols] = metric_display_df[numeric_cols].round(4)
                    render_striped_table(metric_display_df)

                    # Silhouette dan Davies-Bouldin membutuhkan minimal 2 label
                    # berbeda. Dataset satu populasi (mis. Jawa Barat) tidak boleh
                    # dipaksa menjadi beberapa cluster hanya demi menghasilkan skor.
                    n_unique_labels = len(np.unique(labels)) if labels is not None else 0
                    if labels is not None and len(metric_df) > 0:
                        best_trust = metric_df.loc[metric_df["Trustworthiness"].idxmax(), "Metode"]
                        if n_unique_labels >= 2 and n_unique_labels < len(labels):
                            best_sil = metric_df.loc[metric_df["Silhouette (macro-avg)"].idxmax(), "Metode"]
                            best_dbi = metric_df.loc[metric_df["Davies-Bouldin"].idxmin(), "Metode"]
                            st.caption(
                                f"**Ringkasan metrik:** Silhouette macro-avg tertinggi → {best_sil} | "
                                f"DBI terendah → {best_dbi} | "
                                f"Trustworthiness tertinggi → {best_trust}"
                            )
                        else:
                            st.info(
                                "Metadata hanya memiliki satu populasi. Silhouette Score dan "
                                "Davies-Bouldin Index tidak dihitung karena keduanya memerlukan "
                                "minimal dua label berbeda. Trustworthiness tetap dihitung."
                            )
                            st.caption(
                                f"**Ringkasan metrik:** Trustworthiness tertinggi → {best_trust}"
                            )

                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.download_button(
                            "Unduh tabel metrik (.csv)",
                            data=metric_df.to_csv(index=False).encode("utf-8"),
                            file_name="metrik_evaluasi.csv",
                        )

        with right_col:
          with st.container(border=True):
            st.markdown('<div class="section-label">Informasi individu</div>', unsafe_allow_html=True)
            sel = st.session_state.selected_sample
            fields = [
                ("ID sampel", sel["sample_id"] if sel else "—"),
                ("Populasi", sel["population"] if sel else "—"),
                ("Metode", sel["method"] if sel else "—"),
                ("Koordinat", ", ".join(f"{c:.3f}" for c in sel["coords"]) if sel else "—"),
            ]
            for label, value in fields:
                st.markdown(
                    f'<div class="info-field-label">{label}</div>'
                    f'<div class="info-field-value">{value}</div>',
                    unsafe_allow_html=True,
                )
            if sel is None:
                st.info("Klik titik pada scatter plot untuk melihat detail individu di kolom ini.")