# Dashboard Evaluasi Algoritma Manifold Learning pada Data Genom

Dashboard Streamlit untuk memvisualisasikan kemiripan genetik antarindividu dari data SNP
(format VCF) dan membandingkan tiga algoritma reduksi dimensi: **UMAP**, **t-SNE**, dan **MDS**.

Dashboard ini adalah luaran pendukung skripsi **"Evaluasi Algoritma Manifold Learning untuk
Visualisasi Kemiripan Genetik Populasi Manusia"** oleh Miftahul Jannah (NIM 2303126066),
Program Studi Sistem Informasi, FMIPA, Universitas Riau.

## Alur pipeline

Dashboard terdiri atas tiga tab yang mengikuti urutan analisis. Komputasi berada di folder
`pipeline/`; `app.py` hanya mengurus tampilan.

| Tab | Modul | Masukan | Keluaran |
|---|---|---|---|
| Preprocessing | `pipeline/preprocessing.py` | File VCF (`.vcf` / `.vcf.gz`), metadata populasi (CSV) | Feature matrix (individu × SNP) |
| Distance Matrix | `pipeline/distance.py` | Feature matrix dari Tab 1 atau `.npz` unggahan | Distance matrix Euclidean |
| Embedding dan Evaluasi | `pipeline/reduction.py` | Distance matrix, metadata populasi | Embedding 1D/2D/3D, scatter plot, tabel metrik |

### Tahap preprocessing (Tab 1)

1. **Filter biallelic SNP**: membuang indel, SV, dan varian multiallelic.
2. **Filter MAF**: membuang SNP dengan minor allele frequency di bawah ambang (default 0,05).
3. **Filter missing genotype**: membuang SNP yang genotipenya hilang pada satu individu atau
   lebih. Yang dibuang adalah SNP (kolom), sehingga seluruh individu tetap dipertahankan.
   Tidak ada imputasi.
4. **LD-pruning** per kromosom dengan `plink2 --indep-pairwise` (default r² = 0,1,
   window = 1000 varian, step = 1). Window dihitung dalam jumlah varian, bukan kb.
5. **Parsing genotipe** menjadi matriks numerik berisi jumlah alel alternatif (0, 1, 2).

Parameter opsional **batas maksimum SNP** mempertahankan SNP dengan MAF tertinggi jika jumlah
SNP setelah pruning melebihi batas (berguna agar data whole-genome muat di RAM).

### Tab 2 dan Tab 3

- Jarak antarindividu dihitung dengan jarak Euclidean.
- Pada Tab 3, parameter UMAP (`n_neighbors`, `min_dist`), t-SNE (`perplexity`), dan MDS
  (`n_init`) dapat diatur. Random seed dikunci ke **42** agar hasil dapat direproduksi.
- Metrik evaluasi: Silhouette Score, Silhouette macro-average, Davies-Bouldin Index, dan
  Trustworthiness. Silhouette dan Davies-Bouldin hanya dihitung bila metadata memuat minimal
  dua populasi.

## Kebutuhan sistem

- Python 3.10 atau lebih baru (diuji dengan Python 3.12)
- [`bcftools`](https://samtools.github.io/bcftools/) dan [`plink2`](https://www.cog-genomics.org/plink/2.0/)
  terpasang dan dapat dipanggil dari terminal (`bcftools --version`, `plink2 --version`)
- Paket Python pada `requirements.txt`
- Genom referensi **hg19/GRCh37** (VCF berbuild GRCh38 ditolak saat validasi)
- Linux, macOS, atau **Windows dengan WSL2 (Ubuntu)**. `bcftools` dan `plink2` berjalan di
  Linux, sehingga pengguna Windows menjalankan dashboard di WSL2.

## Instalasi

### Langkah 1. Siapkan WSL2 (khusus Windows)

Buka **PowerShell sebagai Administrator**, lalu jalankan:

```powershell
wsl --install -d Ubuntu
```

Restart komputer jika diminta, buka aplikasi **Ubuntu** dari menu Start, lalu buat nama
pengguna dan kata sandi. **Semua perintah berikutnya dijalankan di terminal Ubuntu ini.**
Pengguna Linux atau macOS langsung memakai terminal biasa.

### Langkah 2. Pasang alat dasar, bcftools, dan plink2

```bash
sudo apt-get update
sudo apt-get install -y git unzip python3 python3-venv python3-pip python3-tk bcftools tabix
sudo apt-get install -y plink2
```

Jika perintah terakhir gagal (`Unable to locate package plink2`), pasang `plink2` secara manual:
unduh versi **Linux 64-bit** dari https://www.cog-genomics.org/plink/2.0/, lalu

```bash
unzip plink2_linux_x86_64_*.zip
sudo mv plink2 /usr/local/bin/
sudo chmod +x /usr/local/bin/plink2
```

Periksa pemasangan:

```bash
bcftools --version
plink2 --version
```

> Berkas yang diunduh di Windows dapat diakses dari Ubuntu lewat
> `/mnt/c/Users/NAMA_PENGGUNA/Downloads/`.
> Pengguna macOS: pasang lewat Homebrew, yaitu `brew install bcftools htslib git`, lalu
> unduh `plink2` dari situs resminya.

### Langkah 3. Clone repositori

```bash
cd ~
git clone https://github.com/miftahul131415/genome_dashboard.git
cd NAMA-REPO
```


### Langkah 4. Buat virtual environment dan pasang paket Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Setiap kali membuka terminal baru, aktifkan lagi dengan `source .venv/bin/activate` sebelum
menjalankan dashboard.

## Menjalankan dashboard

Dari folder utama proyek, dengan virtual environment aktif:

```bash
streamlit run app.py
```

Dashboard terbuka di `http://localhost:8501`. Pengguna WSL2 cukup membuka alamat itu di
peramban Windows. Jika port 8501 terpakai:

```bash
streamlit run app.py --server.port 8502
```

## Cara memakai dashboard

### Mencoba cepat dengan data contoh

1. Tab **Preprocessing**: centang **Gunakan data contoh**, lalu klik **Jalankan preprocessing**.
2. Tab **Distance Matrix**: pilih **Pakai hasil Tab 1 (sesi ini)**, lalu klik
   **Hitung distance matrix**.
3. Tab **Embedding dan Evaluasi**: klik **Jalankan visualisasi**.

### Tab 1: Preprocessing

1. Tentukan berkas VCF. Ada dua cara:
   - Centang **Gunakan data contoh** (subset kromosom 22 dari 1000 Genomes beserta metadatanya), atau
   - Isi kolom **Path file VCF (.vcf / .vcf.gz)** dengan lokasi berkas, atau klik **Browse**
     untuk memilihnya lewat dialog. Berkas dibaca langsung dari disk dan **tidak diunggah**,
     sehingga dataset besar tetap bisa dipakai.
2. Unggah **metadata populasi** (CSV) dengan kolom `sample_id` dan `population`. Nilai
   `sample_id` harus sama dengan nama sampel di VCF. Tanpa metadata, Silhouette dan
   Davies-Bouldin tidak dapat dihitung.

   ```csv
   sample_id,population
   HG00096,EUR
   HG00097,EUR
   ```
3. Atur parameter:

   | Parameter | Bawaan |
   |---|---|
   | Ambang MAF | 0,05 |
   | Ambang r² LD-pruning | 0,1 |
   | Window size (jumlah SNP) | 1000 |
   | Step size LD-pruning | 1 (terkunci) |
   | Batas maksimum SNP | 0 (tanpa batas) |

4. Klik **Jalankan preprocessing**. Proses diawali validasi berkas; jika gagal, pesan
   kesalahan ditampilkan.
5. Hasilnya berupa ringkasan (total individu, total SNP setelah pruning, jumlah missing
   genotype) dan tabel jumlah varian pada setiap tahap. Feature matrix dapat diunduh sebagai
   `.npz` (hanya untuk berkas di bawah 300 MB).

### Tab 2: Distance Matrix

1. Pilih **Sumber feature matrix**:
   - **Pakai hasil Tab 1 (sesi ini)**, atau
   - **Upload feature matrix sendiri (.npz)** untuk melanjutkan dari hasil unduhan sebelumnya.
2. Klik **Hitung distance matrix**. Jarak yang dipakai adalah Euclidean dan tidak dapat diganti.
3. Ringkasan hasil menampilkan ukuran matriks, status simetris, dan pemeriksaan diagonal bernilai nol.

### Tab 3: Embedding dan Evaluasi

1. Unggah metadata populasi jika belum ada (`.csv`, atau berkas `.panel` 1000 Genomes dengan
   kolom `sample`, `pop`, `super_pop`; untuk `.panel` pilih level populasi yang dipakai sebagai label).
2. Pilih **Dimensi** (1D, 2D, atau 3D), metode yang ditampilkan (UMAP, t-SNE, MDS), dan
   atur parameter: **UMAP n_neighbors**, **UMAP min_dist**, **t-SNE perplexity**, dan **MDS n_init**.
   Ketiga metode selalu dihitung bersama; kotak centang hanya mengatur tampilan.
3. Klik **Jalankan visualisasi**.
4. Gunakan filter populasi dan kolom **Cari ID sampel** untuk menyaring titik. Pada grafik
   interaktif, perbesar, geser, dan pilih titik (tampilan 1D dan 2D) untuk melihat ID sampel,
   populasi, metode, dan koordinatnya. Pada tampilan 3D, informasi muncul lewat hover.
5. Tabel metrik di bawah grafik dapat diunduh lewat **Unduh tabel metrik (.csv)**.
6. **Reset sesi** menghapus hasil yang tersimpan agar analisis dapat dimulai dari awal.

## Data

| File | Isi |
|---|---|
| `sample_data/chr22_sample.vcf.gz` | Subset kromosom 22 dari 1000 Genomes Phase 3 (2.504 individu, belum difilter), untuk demo |
| `sample_data/metadata_1000genomes.csv` | Label populasi 1000 Genomes (`sample_id`, `population`) |
| `sample_data/westjava_metadata.csv` | Label populasi dataset WestJava (data genotipenya tidak disertakan di repositori) |

Dataset lengkap yang dipakai dalam penelitian, dalam format VCF dengan genom referensi
GRCh37/hg19:

- **1000 Genomes Project Phase 3**: International Genome Sample Resource (IGSR),
  https://www.internationalgenome.org/
- **WestJava Whole Genome Sequencing Dataset**: European Nucleotide Archive (ENA), kode
  proyek PRJEB85629

Format metadata CSV: kolom wajib `sample_id` dan `population`.

## Struktur proyek

```
.
├── app.py                  # Antarmuka Streamlit (tiga tab)
├── pipeline/
│   ├── preprocessing.py    # Tab 1: filter, LD-pruning, feature matrix
│   ├── distance.py         # Tab 2: distance matrix Euclidean
│   ├── reduction.py        # Tab 3: UMAP, t-SNE, MDS, metrik evaluasi
│   └── cache_utils.py      # Pembuatan cache key
├── sample_data/            # Data contoh dan metadata
├── cache/                  # Hasil antara (dibuat otomatis, tidak di-commit)
├── .streamlit/config.toml
└── requirements.txt
```

## Cache

Hasil setiap tahap disimpan di folder `cache/`. Jika file input dan parameternya tidak berubah,
tahap tersebut tidak dihitung ulang. Folder ini dapat dihapus kapan saja dan akan dibuat ulang
secara otomatis. Untuk data whole-genome, ukurannya bisa mencapai beberapa GB.

## Catatan penting

- **Lokasi data (WSL2):** simpan berkas VCF di dalam sistem berkas Linux (misalnya `~/data/`),
  bukan di `/mnt/c/` atau `/mnt/d/`. Pemrosesan di folder Windows jauh lebih lambat.
- **Format path (WSL2):** pakai path Linux, misalnya `/home/nama/data/file.vcf.gz`. Path Windows
  seperti `C:\Users\...` tidak dikenali oleh `bcftools` dan `plink2`.
- **Tombol Browse:** butuh `python3-tk` dan tampilan desktop. Jika tidak tersedia, ketik path
  secara manual.
- **Batas unggah `.npz`:** Streamlit membatasi unggahan 200 MB secara bawaan. Untuk berkas lebih
  besar, jalankan `streamlit run app.py --server.maxUploadSize 2048`, atau jalankan ulang Tab 1
  (cepat karena ada cache).
- **Proses berhenti di tengah:** jalankan ulang dengan parameter yang sama. Tahap yang sudah
  selesai dilanjutkan dari cache, termasuk LD-pruning per kromosom.

## Pemecahan masalah

| Pesan atau gejala | Penyebab dan solusi |
|---|---|
| `Tool berikut tidak ditemukan di PATH: bcftools, plink2` | Alat belum terpasang atau belum ada di PATH. Ulangi Langkah 2, lalu jalankan dashboard dari terminal yang sama. |
| `File tidak ditemukan` | Path salah. Pengguna WSL2 harus memakai path Linux. |
| `File harus berformat .vcf atau .vcf.gz (bgzip)` | Format berkas tidak sesuai. |
| `Build genom terdeteksi bukan hg19/GRCh37` | Berkas memakai GRCh38. Gunakan berkas GRCh37/hg19. |
| `Tidak ada sampel (kolom genotipe) di file VCF ini` | VCF tidak memiliki kolom genotipe. Periksa berkasnya. |
| `command not found: streamlit` | Virtual environment belum aktif. Jalankan `source .venv/bin/activate`. |
| Dialog Browse tidak muncul | Ketik path secara manual (lihat Catatan penting). |
| Proses berhenti karena memori habis | Tutup aplikasi lain, atau isi **Batas maksimum SNP** pada Tab 1. |

## Lingkungan pengujian

Revisi terakhir diuji dengan Python 3.12, bcftools 1.19, plink2 v2.00a6, Streamlit 1.67,
scikit-allel 1.3.13, umap-learn 0.5.12, dan scikit-learn 1.9.1. Untuk mencatat versi pada
lingkungan Anda sendiri, jalankan `pip freeze`.
