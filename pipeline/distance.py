"""
Pipeline Tab 2 - Hitung Jarak.

Metrik jarak dikunci ke Euclidean sesuai batasan masalah penelitian, sehingga
tidak ada parameter metrik yang dapat diubah pengguna di sini.
"""
from pathlib import Path

import numpy as np

from .cache_utils import make_cache_key

COLUMN_BATCH = 2000  # jumlah SNP per batch (batch float64 ~ n_sampel x 2000 x 8 byte)

def euclidean_distance_matrix(feature_matrix, column_batch: int = COLUMN_BATCH) -> np.ndarray:
    """Distance matrix Euclidean (n_sampel x n_sampel) dari feature matrix
    (n_sampel x n_snp), dihitung per batch kolom."""
    X = feature_matrix
    if X.ndim != 2:
        raise ValueError("Feature matrix harus 2 dimensi (individu x SNP).")
    n, n_snp = X.shape

    gram = np.zeros((n, n), dtype=np.float64)
    for start in range(0, n_snp, column_batch):
        block = np.asarray(X[:, start:start + column_batch], dtype=np.float64)
        gram += block @ block.T

    sq_norm = np.diag(gram).copy()
    # d^2 = |xi|^2 + |xj|^2 - 2<xi,xj>, dikerjakan in-place supaya hemat memori
    gram *= -2.0
    gram += sq_norm[:, None]
    gram += sq_norm[None, :]
    np.maximum(gram, 0.0, out=gram)  # buang selisih negatif akibat pembulatan
    np.sqrt(gram, out=gram)

    # jamin simetri persis & diagonal nol persis
    upper = np.triu(gram, k=1)
    dist = upper + upper.T
    return dist


def compute_distance_matrix(feature_matrix: np.ndarray, workdir: str,
                             source_key: str) -> dict:
    """Hitung distance matrix Euclidean dari feature matrix (n_sample x n_snp).

    source_key: identifier unik dari feature matrix asal (biasanya
    feature_matrix_cache_path dari hasil Tab 1) supaya cache Tab 2 ikut
    berubah kalau Tab 1 di-rerun dengan parameter berbeda.
    """
    Path(workdir).mkdir(parents=True, exist_ok=True)
    key = make_cache_key(
        source_key if Path(source_key).exists() else __file__,
        {"step": "distance", "metric": "euclidean", "impl": "gram_batched_v2",
         "shape": list(feature_matrix.shape),
         "source": str(source_key)},
        prefix="dist",
    )
    out_path = Path(workdir) / f"{key}.npy"

    if out_path.exists():
        dist_matrix = np.load(out_path)
    else:
        dist_matrix = euclidean_distance_matrix(feature_matrix)
        tmp_path = out_path.with_name(out_path.stem + ".tmp.npy")
        try:
            np.save(tmp_path, dist_matrix)
            tmp_path.replace(out_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return {
        "distance_matrix": dist_matrix,
        "cache_path": str(out_path),
        "shape": dist_matrix.shape,
        "is_symmetric": bool(np.allclose(dist_matrix, dist_matrix.T)),
        "diagonal_is_zero": bool(np.allclose(np.diag(dist_matrix), 0)),
    }
