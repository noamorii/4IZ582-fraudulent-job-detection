"""Export the complete embedding dataset, splits and generation statistics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metadata_signature(metadata):
    columns = ["job_id", "fraudulent", "text_hash", "group_id", "split", "cv_fold"]
    return hashlib.sha256(metadata[columns].to_csv(index=False).encode()).hexdigest()


def save_cache(output_dir, vectors, metadata, *, source, is_mock,
               embedding_seconds, endpoint_requests, extra=None):
    """Save endpoint results without calling an API or changing the shared split.

    Each row of vectors must correspond to the same row of metadata. No labels
    or job IDs may have been sent to the embedding model as part of its input.
    endpoint_requests includes actual HTTP attempts (including retries).
    """
    output = Path(output_dir)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(metadata) or vectors.shape[1] < 1:
        raise ValueError("Expected one non-empty embedding vector per metadata row")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite")
    if not isinstance(is_mock, bool) or not source:
        raise ValueError("Specify source and boolean is_mock explicitly")
    if not np.isfinite(embedding_seconds) or embedding_seconds < 0:
        raise ValueError("embedding_seconds must be a non-negative finite number")
    if not isinstance(endpoint_requests, int) or endpoint_requests < 0:
        raise ValueError("endpoint_requests must count actual HTTP attempts")
    output.mkdir(parents=True, exist_ok=True)
    metadata_tmp = output / "metadata.csv.tmp"
    metadata.to_csv(metadata_tmp, index=False)
    metadata_tmp.replace(output / "metadata.csv")
    # A CSV with a JSON array in each embedding cell keeps the vector dimension
    # independent of the model and mirrors the endpoint's list[float] response.
    embeddings_tmp = output / "embeddings.csv.tmp"
    pd.DataFrame({
        "job_id": metadata["job_id"].to_numpy(),
        "text_hash": metadata["text_hash"].to_numpy(),
        "embedding": [json.dumps(v.tolist(), separators=(",", ":")) for v in vectors],
    }).to_csv(embeddings_tmp, index=False)
    embeddings_tmp.replace(output / "embeddings.csv")
    saved_metadata = pd.read_csv(output / "metadata.csv", keep_default_na=False)
    manifest = dict(extra or {})
    manifest.update({
        "source": source, "is_mock": is_mock, "n_rows": len(metadata),
        "dimensions": vectors.shape[1],
        "n_folds": int(saved_metadata.loc[saved_metadata["split"].eq("train"), "cv_fold"].nunique()),
        "metadata_sha256": metadata_signature(saved_metadata),
        "embedding_seconds": float(embedding_seconds), "endpoint_requests": endpoint_requests,
    })
    info_tmp = output / "manifest.json.tmp"
    info_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    info_tmp.replace(output / "manifest.json")
    return manifest
