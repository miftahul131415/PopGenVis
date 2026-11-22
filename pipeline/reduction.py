"""
Pipeline Tab 3 - Reduksi Dimensi & Evaluasi.

Tiga metode selalu dihitung (UMAP, t-SNE, MDS).
Checkbox di UI hanya mengatur metode yang ditampilkan.
Random seed dikunci 42 untuk reproducibility.

Catatan evaluasi:
- Silhouette Score dan Davies-Bouldin Index membutuhkan minimal 2 label
  berbeda dan tidak boleh dipaksakan untuk dataset satu populasi.
- Trustworthiness tetap dapat dihitung tanpa label populasi.
"""
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE, MDS, trustworthiness
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import umap

from .cache_utils import make_cache_key

RANDOM_SEED_LOCKED = 42


def _run_umap(distance_matrix, n_components, n_neighbors, min_dist):
    """UMAP pada distance matrix (metric="precomputed"); n_neighbors dibatasi agar < jumlah sampel."""
    n_samples = distance_matrix.shape[0]
    n_neighbors = max(2, min(int(n_neighbors), n_samples - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric="precomputed",
        random_state=RANDOM_SEED_LOCKED,
    )
    return reducer.fit_transform(distance_matrix)


def _run_tsne(distance_matrix, n_components, perplexity):
    """t-SNE pada distance matrix (metric="precomputed"); perplexity dibatasi agar < jumlah sampel."""
    n_samples = distance_matrix.shape[0]
    max_perplexity = max(2, n_samples - 1)
    perplexity = float(max(2, min(float(perplexity), max_perplexity)))
    # scikit-learn mensyaratkan perplexity < n_samples.
    if perplexity >= n_samples:
        perplexity = max(1.0, n_samples - 1.0)
    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        metric="precomputed",
        init="random",
        random_state=RANDOM_SEED_LOCKED,
    )
    return reducer.fit_transform(distance_matrix)


def _run_mds(distance_matrix, n_components, n_init=4):
    """MDS (SMACOF) dengan matriks jarak yang sudah dihitung.

    n_init = jumlah percobaan SMACOF dari posisi awal acak yang berbeda;
    hasil dengan stress terkecil yang dipakai. Nilai bawaan 4 sama dengan
    konfigurasi utama sebelumnya.
    """
    common = dict(
        n_components=n_components,
        random_state=RANDOM_SEED_LOCKED,
        normalized_stress=False,
        n_init=int(n_init),
    )
    try:
        # scikit-learn >= 1.8: parameter `dissimilarity` diganti `metric`.
        # init="random" dipasang eksplisit karena init="classical_mds"
        # mengabaikan n_init.
        reducer = MDS(metric="precomputed", init="random", **common)
    except TypeError:
        # scikit-learn < 1.8 (belum ada parameter `init` di konstruktor).
        reducer = MDS(dissimilarity="precomputed", **common)
    return reducer.fit_transform(distance_matrix)


def _has_valid_cluster_labels(labels, n_samples):
    """True jika label valid untuk metrik clustering berbasis label."""
    if labels is None:
        return False
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n_samples:
        return False
    n_unique = len(np.unique(labels))
    return 2 <= n_unique <= n_samples - 1


def _silhouette_macro_averaged(embedding, labels):
    """Silhouette macro-average: rata-rata silhouette per kelompok, lalu dirata-ratakan antar kelompok."""
    sample_scores = silhouette_samples(embedding, labels, metric="euclidean")
    labels_arr = np.asarray(labels)
    unique_labels = np.unique(labels_arr)
    per_label_mean = [
        sample_scores[labels_arr == lbl].mean() for lbl in unique_labels
    ]
    return float(np.mean(per_label_mean))


def compute_metrics(distance_matrix: np.ndarray, embedding: np.ndarray, labels) -> dict:
    """Hitung metrik evaluasi tanpa memaksakan clustering pada satu populasi."""
    n_samples = embedding.shape[0]
    valid_labels = _has_valid_cluster_labels(labels, n_samples)

    if valid_labels:
        labels = np.asarray(labels)
        metrics = {
            "silhouette": float(
                silhouette_score(embedding, labels, metric="euclidean")
            ),
            "silhouette_macro": _silhouette_macro_averaged(embedding, labels),
            "davies_bouldin": float(
                davies_bouldin_score(embedding, labels)
            ),
        }
    else:
        # Dataset tanpa label atau hanya satu populasi: metrik clustering
        # berbasis label tidak terdefinisi secara metodologis.
        metrics = {
            "silhouette": None,
            "silhouette_macro": None,
            "davies_bouldin": None,
        }

    if n_samples > 2:
        n_neighbors = min(5, n_samples - 1)
        metrics["trustworthiness"] = float(
            trustworthiness(
                distance_matrix,
                embedding,
                n_neighbors=n_neighbors,
                metric="precomputed",
            )
        )
    else:
        metrics["trustworthiness"] = None

    return metrics


def _cache_embedding(workdir, source_key, method, params, compute_fn):
    """Cache embedding berdasarkan input distance matrix + seluruh parameter."""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    key = make_cache_key(
        source_key,
        {"step": "reduction", "method": method, **params},
        prefix="reduction",
    )
    out_path = Path(workdir) / f"{key}_{method.lower().replace('-', '_')}.npy"
    if out_path.exists():
        return np.load(out_path)
    embedding = compute_fn()
    np.save(out_path, embedding)
    return embedding


def run_reduction_and_evaluation(
    distance_matrix: np.ndarray,
    workdir: str,
    source_key: str,
    labels=None,
    n_components: int = 2,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30,
    mds_n_init: int = 4,
    progress_callback=None,
) -> dict:
    """Jalankan UMAP, t-SNE, MDS dan evaluasi untuk semuanya."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix harus berbentuk matriks persegi.")

    n_samples = distance_matrix.shape[0]
    if n_samples < 3:
        raise ValueError("Reduksi dimensi membutuhkan minimal 3 sampel.")
    if n_components not in (1, 2, 3):
        raise ValueError("n_components harus 1, 2, atau 3.")

    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != n_samples:
            raise ValueError(
                f"Jumlah label ({len(labels) if labels.ndim == 1 else 'tidak valid'}) "
                f"harus sama dengan jumlah sampel ({n_samples})."
            )

    results = {}

    if progress_callback:
        progress_callback("Menghitung UMAP...")
    emb_umap = _cache_embedding(
        workdir,
        source_key,
        "UMAP",
        {
            "n_components": n_components,
            "n_neighbors": min(int(umap_n_neighbors), n_samples - 1),
            "min_dist": float(umap_min_dist),
            "seed": RANDOM_SEED_LOCKED,
        },
        lambda: _run_umap(
            distance_matrix, n_components, umap_n_neighbors, umap_min_dist
        ),
    )
    results["UMAP"] = {
        "embedding": emb_umap,
        "metrics": compute_metrics(distance_matrix, emb_umap, labels),
    }

    if progress_callback:
        progress_callback("Menghitung t-SNE...")
    emb_tsne = _cache_embedding(
        workdir,
        source_key,
        "t-SNE",
        {
            "n_components": n_components,
            "perplexity": float(tsne_perplexity),
            "seed": RANDOM_SEED_LOCKED,
        },
        lambda: _run_tsne(distance_matrix, n_components, tsne_perplexity),
    )
    results["t-SNE"] = {
        "embedding": emb_tsne,
        "metrics": compute_metrics(distance_matrix, emb_tsne, labels),
    }

    if progress_callback:
        progress_callback("Menghitung MDS...")
    emb_mds = _cache_embedding(
        workdir,
        source_key,
        "MDS",
        {
            "n_components": n_components,
            "seed": RANDOM_SEED_LOCKED,
            "n_init": int(mds_n_init),
        },
        lambda: _run_mds(distance_matrix, n_components, mds_n_init),
    )
    results["MDS"] = {
        "embedding": emb_mds,
        "metrics": compute_metrics(distance_matrix, emb_mds, labels),
    }

    if progress_callback:
        progress_callback("Reduksi dimensi dan evaluasi selesai.")

    return results