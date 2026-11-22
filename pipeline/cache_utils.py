"""
Utilitas cache untuk pipeline.

Cache key adalah hash dari (path, ukuran, dan waktu ubah file input) beserta
SEMUA parameter yang mempengaruhi hasil. Jika file input atau salah satu
parameter berubah, key ikut berubah sehingga cache lama tidak terpakai lagi.
Setiap hasil disimpan sebagai file terpisah per key di folder kerja (workdir).
"""
import hashlib
import json
from pathlib import Path


def _file_fingerprint(path: str) -> dict:
    """Ambil sidik jari file (path, ukuran, waktu ubah) tanpa membaca isinya.

    Isi file sengaja tidak di-hash karena VCF whole-genome bisa berukuran
    puluhan GB. Kombinasi path, ukuran, dan waktu ubah cukup untuk mendeteksi
    perubahan pada penggunaan normal.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    stat = p.stat()
    return {"path": str(p.resolve()), "size": stat.st_size, "mtime": stat.st_mtime}


def make_cache_key(input_paths, params: dict, prefix: str = "") -> str:
    """Buat cache key dari satu atau lebih file input dan dict parameter.

    input_paths: path tunggal (str) atau list path.
    params: parameter yang mempengaruhi hasil (harus bisa di-serialisasi ke JSON).
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    fingerprints = [_file_fingerprint(p) for p in input_paths]
    payload = {
        "files": fingerprints,
        "params": params,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:20]
    return f"{prefix}_{digest}" if prefix else digest
