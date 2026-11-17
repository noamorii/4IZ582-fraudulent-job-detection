import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


def get_embeddings(texts, cache_folder, batch_size=64, offline=False):
    """Return vectors and statistics. Reuse cached batches; offline mode never calls the API."""
    base_url, api_key, model_name = load_embedding_settings()
    cache_folder = Path(cache_folder)
    client = None
    embedding_batches = []
    batch_stats = {}

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        request_key = make_request_key(base_url=base_url, model=model_name, texts=batch_texts)
        cache_file = cache_folder / f"{request_key}.json"
        from_cache = cache_file.exists()

        if from_cache:
            batch_vectors, batch_result = load_cached_batch(cache_file, request_key, len(batch_texts))
        else:
            if offline:
                raise FileNotFoundError(f"Missing offline embedding batch: {cache_file}")
            if client is None:
                client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=2)

            batch_vectors, elapsed_seconds, request_count = request_embedding_batch(client, batch_texts, model_name)
            batch_result = {
                "request_key": request_key,
                "model": model_name,
                "base_url": base_url,
                "vectors": batch_vectors.tolist(),
                "endpoint_requests": request_count,
                "embedding_seconds": elapsed_seconds,
            }
            write_json(cache_file, batch_result)

        embedding_batches.append(batch_vectors)
        # Repeated cache entries contribute vectors, but no additional costs.
        if request_key not in batch_stats:
            batch_stats[request_key] = {
                "endpoint_requests": batch_result["endpoint_requests"],
                "embedding_seconds": batch_result["embedding_seconds"],
                "new_endpoint_requests": 0 if from_cache else batch_result["endpoint_requests"],
            }

    stats = {
        **summarize_batch_stats(batch_stats),
        "embedding_model": model_name,
        "base_url": base_url,
        "batch_size": batch_size,
    }
    return np.concatenate(embedding_batches), stats


def save_embedding_dataset(vectors, metadata, stats, output_folder):
    """Save embeddings.csv, metadata.csv and manifest.json; return the manifest.

    Vector row i must belong to metadata row i. Statistics come from
    get_embeddings(); exporting never calls the API or changes the splits.
    """
    vectors = validate_vectors(vectors, len(metadata))

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    embedding_table = pd.DataFrame({
        "job_id": metadata["job_id"].to_numpy(),
        "text_hash": metadata["text_hash"].to_numpy(),
        "embedding": [json.dumps(vector.tolist(), separators=(",", ":")) for vector in vectors],
    })

    # Write each table completely before replacing its previous version.
    for filename, table in [("metadata.csv", metadata), ("embeddings.csv", embedding_table)]:
        temporary_file = output_folder / f"{filename}.tmp"
        table.to_csv(temporary_file, index=False)
        temporary_file.replace(output_folder / filename)

    # Sign the saved representation so the loader can verify the same rows/splits.
    saved_metadata = pd.read_csv(output_folder / "metadata.csv", keep_default_na=False)
    manifest = {
        **stats,
        "source": stats.get("source", f"school/{stats['embedding_model']}"),
        "is_mock": stats.get("is_mock", False),
        "n_rows": len(metadata),
        "dimensions": vectors.shape[1],
        "n_folds": int(saved_metadata.loc[saved_metadata["split"].eq("train"), "cv_fold"].nunique()),
        "metadata_sha256": metadata_hash(saved_metadata),
        "embedding_seconds": float(stats["embedding_seconds"]),
    }
    write_json(output_folder / "manifest.json", manifest)
    return manifest


def load_embedding_settings():
    # Read .env only on an explicit call; importing this module does no I/O.
    from dotenv import load_dotenv
    load_dotenv()
    return (
        os.environ.get("LLM_BASE_URL", os.environ.get("BASE_URL", "https://litellm.vse.cz/v1/")),
        os.environ.get("LLM_API_KEY", os.environ.get("API_KEY", "")).strip(),
        os.environ.get("LLM_EMBED_MODEL", "qwen3-embedding:4b"),
    )


def make_request_key(**request):
    """Hash the endpoint, model and ordered texts to identify a cached request."""
    serialized_request = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized_request.encode("utf-8")).hexdigest()


def write_json(path, value):
    """Write JSON through a temporary file, then replace the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def validate_vectors(vectors, n_rows, indices=None):
    """Check vector values and, for API responses, their input indices."""
    if indices is not None and list(indices) != list(range(n_rows)):
        raise ValueError("Embedding response has missing or duplicate input indices")

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != n_rows or vectors.shape[1] < 1:
        raise ValueError("Expected one non-empty embedding vector per input text")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinity")
    return vectors


def load_cached_batch(path, expected_request_key, text_count):
    """Read and validate one JSON batch from the request cache."""
    batch_result = json.loads(path.read_text(encoding="utf-8"))
    if batch_result.get("request_key") != expected_request_key:
        raise ValueError(f"Invalid cache entry: {path}")

    vectors = validate_vectors(batch_result["vectors"], text_count)
    return vectors, batch_result


def request_embedding_batch(client, texts, model_name):
    """Request one batch and return vectors, elapsed seconds and HTTP attempt count."""
    started_at = time.perf_counter()
    raw_response = client.embeddings.with_raw_response.create(
        model=model_name, input=texts, encoding_format="float",
    )
    response = raw_response.parse()
    elapsed_seconds = time.perf_counter() - started_at

    # Restore input order before validation.
    response_items = sorted(response.data, key=lambda item: item.index)
    vectors = validate_vectors(
        [item.embedding for item in response_items],
        len(texts),
        indices=[item.index for item in response_items],
    )
    return vectors, elapsed_seconds, 1 + raw_response.retries_taken


def summarize_batch_stats(batch_stats):
    """Sum generation costs once per unique batch, including cached results."""
    return {
        "endpoint_requests": sum(batch["endpoint_requests"] for batch in batch_stats.values()),
        "embedding_seconds": sum(batch["embedding_seconds"] for batch in batch_stats.values()),
        "new_endpoint_requests": sum(batch["new_endpoint_requests"] for batch in batch_stats.values()),
    }


def metadata_hash(metadata):
    """Return a SHA-256 hash of the rows, labels, groups and split assignments."""
    columns = ["job_id", "fraudulent", "text_hash", "group_id", "split", "cv_fold"]
    return hashlib.sha256(metadata[columns].to_csv(index=False).encode()).hexdigest()
