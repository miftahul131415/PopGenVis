"""
Pipeline Tab 1 - Preprocessing.

Urutan tahap:
1. Filter biallelic SNP (bcftools): membuang indel, SV, dan varian multiallelic
   agar genotipe dapat dikodekan sebagai 0/1/2.
2. Filter MAF (bcftools): default 0,05, dapat diubah dari dashboard.
3. Filter missing genotype (bcftools): membuang SNP yang genotipenya hilang
   pada satu individu atau lebih. Yang dibuang adalah SNP (kolom), bukan
   individu (baris), sehingga seluruh individu tetap dipertahankan.
4. LD-pruning per kromosom (plink2 --indep-pairwise), lalu hasil tiap
   kromosom digabung dengan `bcftools concat`.
5. Parsing genotipe menjadi matriks numerik 0/1/2 (scikit-allel), dibaca per
   batch varian agar hemat memori.

Setiap tahap dapat dipanggil terpisah. Input boleh .vcf (polos) atau .vcf.gz
(bgzip). VCF polos dibaca secara streaming oleh tahap pertama dan hasilnya
ditulis sebagai .vcf.gz terindeks di workdir, sehingga tahap-tahap berikutnya
selalu bekerja dengan .vcf.gz.

Catatan implementasi:
- Hasil setiap tahap ditulis ke file sementara lalu di-rename. Dengan begitu
  proses yang terhenti di tengah jalan tidak meninggalkan file setengah jadi
  yang dianggap valid oleh cache pada run berikutnya.
- LD-pruning berjalan per kromosom dan hasilnya disimpan sebagai file bagian,
  sehingga run whole-genome yang terhenti dapat dilanjutkan dari kromosom
  yang belum selesai.
- Window LD-pruning dinyatakan dalam JUMLAH VARIAN (bukan kb), mengikuti
  satuan `-w` pada `bcftools +prune` bila ditulis tanpa suffix.
"""
import os
import subprocess
import shutil
from pathlib import Path

import numpy as np
import allel

from .cache_utils import make_cache_key

MAF_DEFAULT = 0.05  # nilai pada eksperimen utama; pengguna dapat mengubahnya di dashboard


class PreprocessingError(Exception):
    """Error dengan pesan yang dapat langsung ditampilkan ke pengguna di UI."""


# ------------------------------------------------------------------
# Helper internal
# ------------------------------------------------------------------
def _run(cmd: list, step_name: str):
    """Jalankan perintah eksternal; ubah kegagalannya menjadi PreprocessingError."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PreprocessingError(
            f"Gagal pada tahap '{step_name}'.\nPerintah: {' '.join(cmd)}\n"
            f"Pesan error:\n{result.stderr[-2000:]}"
        )
    return result


def _check_tools():
    """Pastikan tool eksternal yang wajib tersedia di PATH."""
    missing = [t for t in ("bcftools", "plink2") if shutil.which(t) is None]
    if missing:
        raise PreprocessingError(
            f"Tool berikut tidak ditemukan di PATH: {', '.join(missing)}. "
            "Pasang dulu, lalu jalankan ulang dashboard dari terminal yang sama."
        )


def _has_index(path) -> bool:
    """True jika file VCF memiliki index (.tbi atau .csi)."""
    p = str(path)
    return Path(p + ".tbi").exists() or Path(p + ".csi").exists()


def _is_ready(path) -> bool:
    """File hasil tahap dianggap siap dipakai (cache valid) jika file-nya ada
    DAN sudah punya index."""
    return Path(path).exists() and _has_index(path)


def _count_variants(path) -> int:
    """Hitung jumlah varian tanpa shell (aman untuk path yang mengandung spasi).

    Untuk .vcf.gz yang sudah terindeks dipakai `bcftools index -n` (instan).
    Selain itu file dibaca streaming dengan `bcftools view -H` dan barisnya
    dihitung.
    """
    path = str(path)
    if path.endswith(".gz") and _has_index(path):
        res = subprocess.run(
            ["bcftools", "index", "-n", path], capture_output=True, text=True
        )
        if res.returncode == 0:
            try:
                return int(res.stdout.strip())
            except ValueError:
                pass  # jatuh ke metode streaming

    proc = subprocess.Popen(
        ["bcftools", "view", "-H", path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    n = 0
    for _ in proc.stdout:
        n += 1
    proc.stdout.close()
    if proc.wait() != 0:
        raise PreprocessingError(f"Gagal menghitung jumlah varian dari: {path}")
    return n


def _run_to_tmp_then_move(cmd_builder, out_path: Path, step_name: str,
                          index_step_name: str):
    """Jalankan perintah yang menulis ke file sementara, rename ke out_path
    jika sukses, lalu buat index. cmd_builder(tmp_path) mengembalikan list
    perintah."""
    tmp_path = out_path.with_name(out_path.name.replace(".vcf.gz", ".tmp.vcf.gz"))
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        _run(cmd_builder(str(tmp_path)), step_name)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    _run(["bcftools", "index", "-t", str(out_path)], index_step_name)


# ------------------------------------------------------------------
# Validasi
# ------------------------------------------------------------------
def validate_vcf(vcf_path: str) -> dict:
    """Validasi dasar: format file, build genom, dan keberadaan sampel.

    Menerima .vcf maupun .vcf.gz. Mengembalikan dict info jika valid, atau
    melempar PreprocessingError jika tidak lolos (pesan siap ditampilkan
    sebagai error di UI).
    """
    if not vcf_path or not str(vcf_path).strip():
        raise PreprocessingError("Path file VCF masih kosong.")

    _check_tools()

    # buang spasi/kutip di ujung (umum saat path disalin lewat "Copy as path" Windows)
    p = Path(str(vcf_path).strip().strip('"').strip("'"))
    if not p.exists():
        raise PreprocessingError(f"File tidak ditemukan: {p}")

    name = p.name.lower()
    if not (name.endswith(".vcf.gz") or name.endswith(".vcf")):
        raise PreprocessingError("File harus berformat .vcf atau .vcf.gz (bgzip).")

    header_res = subprocess.run(
        ["bcftools", "view", "-h", str(p)], capture_output=True, text=True)
    if header_res.returncode != 0:
        raise PreprocessingError("File tidak bisa dibaca sebagai VCF yang valid.\n"
            + header_res.stderr[-1000:])
    header = header_res.stdout

    # Panjang kromosom 1 pada header dipakai untuk mengenali build genom.
    grch37_len = "249250621"
    grch38_len = "248956422"
    if grch38_len in header and grch37_len not in header:
        raise PreprocessingError("Build genom terdeteksi bukan hg19/GRCh37, tidak didukung saat ini.")

    samples_res = subprocess.run(
        ["bcftools", "query", "-l", str(p)], capture_output=True, text=True)
    n_samples = samples_res.stdout.count("\n")
    if n_samples == 0:
        raise PreprocessingError("Tidak ada sampel (kolom genotipe) di file VCF ini.")

    n_variants = _count_variants(p)

    return {"n_samples": n_samples, "n_variants_raw": n_variants, "path": str(p)}


# ------------------------------------------------------------------
# Tahap-tahap preprocessing
# ------------------------------------------------------------------
def filter_biallelic_snp(vcf_path: str, workdir: str) -> str:
    """Sisakan biallelic SNP saja. Input boleh .vcf atau .vcf.gz; output selalu
    .vcf.gz terindeks."""
    key = make_cache_key(vcf_path, {"step": "biallelic_snp"}, prefix="snp")
    out_path = Path(workdir) / f"{key}.vcf.gz"
    if _is_ready(out_path):
        return str(out_path)

    _run_to_tmp_then_move(
        lambda tmp: ["bcftools", "view", "--threads", "4",
                     "-v", "snps", "-m2", "-M2", "-Oz", "-o", tmp, vcf_path],
        out_path, "filter biallelic SNP", "index hasil filter SNP",
    )
    return str(out_path)


def filter_maf(vcf_path: str, workdir: str, maf: float = MAF_DEFAULT) -> str:
    """Buang SNP yang minor allele frequency (MAF)-nya di bawah ambang.

    Nilai default 0,05 dipakai pada eksperimen utama, tetapi ambang tidak
    dikunci agar dashboard dapat dipakai untuk analisis dengan parameter lain.
    Secara definisi MAF berada pada rentang 0 sampai 0,5.
    """
    maf = float(maf)
    if not 0.0 <= maf <= 0.5:
        raise PreprocessingError("Ambang MAF harus berada pada rentang 0 sampai 0,5.")
    key = make_cache_key(vcf_path, {"step": "maf", "maf": maf}, prefix="maf")
    out_path = Path(workdir) / f"{key}.vcf.gz"
    if _is_ready(out_path):
        return str(out_path)

    _run_to_tmp_then_move(
        lambda tmp: ["bcftools", "view", "--threads", "4",
                     "-q", f"{maf}:minor", "-Oz", "-o", tmp, vcf_path],
        out_path, "MAF filtering", "index hasil MAF filter",
    )
    return str(out_path)


def filter_missing(vcf_path: str, workdir: str) -> str:
    key = make_cache_key(vcf_path, {"step": "filter_missing"}, prefix="complete")
    out_path = Path(workdir) / f"{key}.vcf.gz"
    if _is_ready(out_path):
        return str(out_path)

    _run_to_tmp_then_move(
        lambda tmp: ["bcftools", "view", "--threads", "4",
                     "-e", 'GT="mis"', "-Oz", "-o", tmp, vcf_path],
        out_path, "filter missing genotype", "index hasil filter missing",
    )
    return str(out_path)


def _list_contigs(vcf_path: str) -> list:
    """Daftar kromosom yang punya record di VCF terindeks (urutan sesuai index)."""
    res = _run(["bcftools", "index", "-s", str(vcf_path)], "daftar kromosom dari index")
    contigs = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].strip() not in ("0", ""):
            contigs.append(parts[0])
    return contigs


def _safe_name(name: str) -> str:
    """Ubah nama kromosom menjadi string aman untuk nama file."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))


def _plink_prune_one(src_vcf: str, prefix: Path, r2: float, window_variants: int,
                     step: int):
    """plink2 --indep-pairwise pada SATU VCF, lalu ekspor varian yang lolos ke
    <prefix>_pruned.vcf.gz. Mengembalikan path hasil, atau None jika 0 SNP tersisa."""
    _run(
        [
            "plink2", "--vcf", src_vcf, "--allow-extra-chr",
            "--set-all-var-ids", "@:#:$r:$a",
            "--new-id-max-allele-len", "1000",
            # tanpa suffix "kb" -> window dalam satuan jumlah varian
            "--indep-pairwise", str(window_variants), str(step), str(r2),
            "--out", str(prefix),
        ],
        "LD-pruning (plink2 --indep-pairwise)",
    )

    prune_in = Path(f"{prefix}.prune.in")
    if not prune_in.exists():
        raise PreprocessingError(
            "plink2 tidak menghasilkan file .prune.in. Cek parameter r²/window."
        )
    with open(prune_in) as f:
        n_kept = sum(1 for _ in f)
    if n_kept == 0:
        return None

    _run(
        [
            "plink2", "--vcf", src_vcf, "--allow-extra-chr",
            "--set-all-var-ids", "@:#:$r:$a",
            "--new-id-max-allele-len", "1000",
            "--extract", str(prune_in),
            "--export", "vcf", "bgz",
            "--out", str(prefix) + "_pruned",
        ],
        "ekstrak SNP hasil pruning",
    )

    pruned_vcf_gz = Path(f"{prefix}_pruned.vcf.gz")
    if not pruned_vcf_gz.exists():
        # Beberapa versi plink2 menulis output tanpa ekstensi .gz walau memakai
        # `--export vcf bgz`. Dalam kasus itu file dikompres sendiri (butuh bgzip).
        alt = Path(f"{prefix}_pruned.vcf")
        if alt.exists():
            _run(["bgzip", str(alt)], "kompres hasil pruning")
            pruned_vcf_gz = Path(str(alt) + ".gz")
        else:
            raise PreprocessingError("File hasil ekstraksi pruning tidak ditemukan.")
    return pruned_vcf_gz


def _remove_quietly(*paths):
    """Hapus file bila ada; tidak error bila file tidak ditemukan."""
    for p in paths:
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass


def ld_prune(vcf_path: str, workdir: str, r2: float = 0.1, window_variants: int = 1000,
             step: int = 1, progress_callback=None) -> str:
    
    key = make_cache_key(
        vcf_path,
        {"step": "ld_prune", "r2": r2, "window_variants": window_variants, "step_size": step},
        prefix="ldprune",
    )
    out_vcf = Path(workdir) / f"{key}.vcf.gz"
    if _is_ready(out_vcf):
        return str(out_vcf)

    contigs = _list_contigs(vcf_path)
    if not contigs:
        raise PreprocessingError("Tidak ada varian yang tersisa sebelum LD-pruning.")
    single = len(contigs) == 1

    part_files = []
    for i, contig in enumerate(contigs, start=1):
        if progress_callback and not single:
            progress_callback(f"LD-pruning kromosom {contig} ({i}/{len(contigs)})...")

        part = Path(workdir) / f"{key}_part{i:03d}_{_safe_name(contig)}.vcf.gz"
        if _is_ready(part):  # sudah selesai pada run sebelumnya
            part_files.append(part)
            continue

        prefix = Path(workdir) / f"{key}_c{i:03d}"
        subset = Path(workdir) / f"{key}_c{i:03d}_subset.vcf.gz"
        try:
            if single:
                src = str(vcf_path)
            else:
                _run(["bcftools", "view", "-r", str(contig), "-Oz", "-o", str(subset),
                      str(vcf_path)], f"ambil kromosom {contig}")
                src = str(subset)

            pruned = _plink_prune_one(src, prefix, r2, window_variants, step)
            if pruned is None:  # kromosom ini tidak menyisakan SNP -> lewati
                continue
            os.replace(pruned, part)
            _run(["bcftools", "index", "-t", str(part)], f"index hasil pruning {contig}")
            part_files.append(part)
        finally:
            for f in Path(workdir).glob(f"{prefix.name}.*"):
                _remove_quietly(f)
            for f in Path(workdir).glob(f"{prefix.name}_pruned.*"):
                _remove_quietly(f)
            _remove_quietly(subset, str(subset) + ".tbi")

    if not part_files:
        raise PreprocessingError(
            "LD-pruning menyisakan 0 SNP. Coba naikkan ambang r² atau perkecil window."
        )

    if len(part_files) == 1:
        os.replace(part_files[0], out_vcf)
        _remove_quietly(str(part_files[0]) + ".tbi")
        _run(["bcftools", "index", "-t", str(out_vcf)], "index hasil akhir pruning")
    else:
        if progress_callback:
            progress_callback("Menggabungkan hasil pruning semua kromosom...")
        _run_to_tmp_then_move(
            lambda tmp: ["bcftools", "concat", "--threads", "4", "-Oz", "-o", tmp]
                        + [str(p) for p in part_files],
            out_vcf, "gabung hasil pruning per kromosom", "index hasil akhir pruning",
        )
        for p in part_files:
            _remove_quietly(p, str(p) + ".tbi")
    return str(out_vcf)


# ------------------------------------------------------------------
# Parsing genotipe -> matriks numerik
# ------------------------------------------------------------------
def _chunk_size(n_samples: int) -> int:
    """Jumlah varian per batch, dijaga agar satu batch GT tetap kecil (puluhan MB)."""
    return int(max(500, min(5000, 8_000_000 // max(int(n_samples), 1))))


def _open_alt_chunks(vcf_path: str, chunk_variants: int):
    """Baca VCF per BATCH VARIAN (baris), bukan sekaligus.

    Mengembalikan (sample_ids, generator). Generator menghasilkan array int8
    berukuran (n_varian_batch, n_sampel) berisi jumlah alel alternatif (0/1/2);
    nilai -1 menandakan genotipe missing.
    """
    _, samples, _, it = allel.iter_vcf_chunks(
        vcf_path, fields=["calldata/GT"], chunk_length=chunk_variants
    )

    def gen():
        for item in it:
            chunk = item[0]
            gt = allel.GenotypeArray(chunk["calldata/GT"])
            yield gt.to_n_alt(fill=-1)

    return [str(s) for s in samples], gen()


def _assert_complete(alt: np.ndarray):
    """Pastikan batch tidak mengandung genotipe missing (nilai -1).

    Setelah filter_missing, kondisi ini selalu terpenuhi. Pengecekan ini
    menjaga agar nilai -1 tidak masuk diam-diam ke feature matrix jika
    parse_genotypes dipanggil pada VCF yang belum melalui filter_missing.
    """
    if (alt < 0).any():
        raise PreprocessingError(
            "Ditemukan genotipe missing pada VCF yang akan diparsing. "
            "Jalankan filter_missing terlebih dahulu."
        )


def _collect_maf(vcf_path: str, chunk_variants: int) -> np.ndarray:
    """Pass 1 (hanya bila ada batas max_snps): hitung MAF setiap varian, per batch."""
    _, gen = _open_alt_chunks(vcf_path, chunk_variants)
    parts = []
    for alt in gen:
        _assert_complete(alt)
        af = alt.sum(axis=1, dtype=np.float64) / alt.shape[1] / 2.0
        parts.append(np.minimum(af, 1.0 - af))
    return np.concatenate(parts) if parts else np.zeros(0)


def parse_genotypes(vcf_path: str, workdir: str, max_snps=None) -> dict:
    """Parsing genotipe menjadi feature matrix numerik (individu x SNP).

    Nilai matriks adalah jumlah alel alternatif (0/1/2). VCF dibaca per batch
    varian dengan scikit-allel (iter_vcf_chunks); matriks hasil dialokasikan
    sekali (float32) lalu diisi batch demi batch. VCF masukan harus sudah
    bebas missing genotype (lihat filter_missing).

    max_snps (opsional): batas maksimum jumlah SNP. Jika jumlah SNP hasil
    pruning melebihi batas, yang dipertahankan adalah SNP dengan MAF tertinggi
    (varian dengan variasi terbesar), dengan urutan posisi aslinya tetap
    dijaga. Ini membutuhkan dua kali baca VCF (pass 1: hitung MAF, pass 2: isi
    matriks); VCF hasil pruning berukuran kecil sehingga biayanya murah.
    """
    max_snps = int(max_snps) if max_snps else 0
    key = make_cache_key(
        vcf_path,
        {"step": "parse_genotypes", "impl": "chunked_f32_v3", "max_snps": max_snps},
        prefix="matrix",
    )
    out_path = Path(workdir) / f"{key}.npz"
    if out_path.exists():
        data = np.load(out_path, allow_pickle=True)
        return {
            "feature_matrix": data["feature_matrix"],
            "sample_ids": data["sample_ids"].tolist(),
            "n_variants_before_cap": int(data["n_variants_before_cap"]),
            "cache_path": str(out_path),
        }

    n_variants = _count_variants(vcf_path)
    if n_variants == 0:
        raise PreprocessingError("Tidak ada varian untuk diparsing.")
    n_samples = len(allel.read_vcf_headers(vcf_path).samples)
    chunk_variants = _chunk_size(n_samples)

    selected = None
    n_out = n_variants
    if max_snps and max_snps < n_variants:
        mafs = _collect_maf(vcf_path, chunk_variants)
        if len(mafs) != n_variants:
            raise PreprocessingError(
                f"Jumlah varian hasil baca ({len(mafs)}) berbeda dari index ({n_variants})."
            )
        top = np.argsort(-mafs, kind="stable")[:max_snps]
        selected = np.zeros(n_variants, dtype=bool)
        selected[top] = True
        n_out = max_snps

    sample_ids, gen = _open_alt_chunks(vcf_path, chunk_variants)
    feature_matrix = np.empty((len(sample_ids), n_out), dtype=np.float32)
    col = 0
    offset = 0
    for alt in gen:
        _assert_complete(alt)
        v = alt.shape[0]
        if selected is not None:
            alt = alt[selected[offset:offset + v]]
        offset += v
        if alt.shape[0] == 0:
            continue
        feature_matrix[:, col:col + alt.shape[0]] = alt.astype(np.float32).T
        col += alt.shape[0]

    if offset != n_variants or col != n_out:
        raise PreprocessingError(
            f"Jumlah varian hasil baca ({offset}) berbeda dari index ({n_variants})."
        )

    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    try:
        np.savez(
            tmp_path,
            feature_matrix=feature_matrix,
            sample_ids=np.array(sample_ids, dtype=object),
            n_variants_before_cap=n_variants,
        )
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "feature_matrix": feature_matrix,
        "sample_ids": sample_ids,
        "n_variants_before_cap": n_variants,
        "cache_path": str(out_path),
    }


# ------------------------------------------------------------------
# Orkestrasi seluruh Tab 1
# ------------------------------------------------------------------
def run_full_preprocessing(vcf_path: str, workdir: str, r2: float = 0.1,
                            window_variants: int = 1000, step: int = 1,
                            maf: float = MAF_DEFAULT, max_snps=None,
                            progress_callback=None) -> dict:
    """Jalankan seluruh Tab 1 secara berurutan.

    progress_callback(str) bersifat opsional dan dipanggil di setiap tahap
    untuk memperbarui status di UI. Validasi VCF (validate_vcf) sudah
    dipanggil di sini, sehingga pemanggil tidak perlu memanggilnya lagi.

    r2, window_variants, step: parameter LD-pruning. window_variants dalam
    JUMLAH VARIAN (bukan kb); lihat ld_prune().
    maf: ambang minor allele frequency (default 0,05; dapat diubah).
    max_snps: batas maksimum jumlah SNP akhir (None/0 = tanpa batas). Jika
    terlampaui, dipertahankan SNP dengan MAF tertinggi (lihat parse_genotypes).

    Mengembalikan dict berisi jumlah varian pada setiap tahap, feature matrix,
    ID sampel, dan path file hasil.
    """
    Path(workdir).mkdir(parents=True, exist_ok=True)

    def _report(msg):
        if progress_callback:
            progress_callback(msg)

    _report("Validasi file VCF...")
    info = validate_vcf(vcf_path)
    vcf_path = info["path"]  # path yang sudah dibersihkan (spasi/kutip di ujung)

    _report("Filter biallelic SNP...")
    snp_only = filter_biallelic_snp(vcf_path, workdir)
    n_after_snp = _count_variants(snp_only)

    _report(f"MAF filtering ({maf*100:g}%)...")
    maf_filtered = filter_maf(snp_only, workdir, maf=maf)
    n_after_maf = _count_variants(maf_filtered)

    _report("Filter missing genotype...")
    complete = filter_missing(maf_filtered, workdir)
    n_after_missing = _count_variants(complete)
    if n_after_missing == 0:
        raise PreprocessingError(
            "Seluruh SNP memiliki missing genotype, sehingga tidak ada SNP yang tersisa "
            "setelah filter missing."
        )

    _report(f"LD-pruning (r²={r2}, window={window_variants} varian, step={step})...")
    pruned = ld_prune(complete, workdir, r2=r2, window_variants=window_variants,
                      step=step, progress_callback=progress_callback)
    n_after_prune = _count_variants(pruned)

    _report("Parsing genotipe (per batch)...")
    parsed = parse_genotypes(pruned, workdir, max_snps=max_snps)

    return {
        "n_samples": info["n_samples"],
        "n_variants_raw": info["n_variants_raw"],
        "n_after_biallelic_snp": n_after_snp,
        "n_after_maf": n_after_maf,
        "maf_threshold": float(maf),
        "n_after_missing_filter": n_after_missing,
        "n_removed_missing": n_after_maf - n_after_missing,
        "n_after_ld_prune": n_after_prune,
        "n_after_cap": int(parsed["feature_matrix"].shape[1]),
        "max_snps": int(max_snps) if max_snps else 0,
        "feature_matrix": parsed["feature_matrix"],
        "sample_ids": parsed["sample_ids"],
        "feature_matrix_cache_path": parsed["cache_path"],
        "pruned_vcf_path": pruned,
    }
