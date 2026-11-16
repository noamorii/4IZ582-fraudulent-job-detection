import hashlib
import json
import os
from pathlib import Path
import time
from openai import OpenAI

import numpy as np


def settings():
    # Read .env only on an explicit call; importing this module does no I/O.
    from dotenv import load_dotenv
    load_dotenv()
    return (
        os.environ.get("LLM_BASE_URL", os.environ.get("BASE_URL", "https://litellm.vse.cz/v1/")),
        os.environ.get("LLM_API_KEY", os.environ.get("API_KEY", "")).strip(),
        os.environ.get("LLM_EMBED_MODEL", "qwen3-embedding:4b"),
    )


def cache_key(**request):
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_cache(path, value):
    """Save an embedding batch to the JSON cache via a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def validate_vectors(vectors, n_rows, indices=None):
    """Check vector values and, for API responses, their input indices."""
    if indices is not None and list(indices) != list(range(n_rows)):
        raise ValueError("Embedding response has missing or duplicate input indices")

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != n_rows or arr.shape[1] < 1:
        raise ValueError("Expected one non-empty embedding vector per input text")
    if not np.isfinite(arr).all():
        raise ValueError("Embeddings contain NaN or infinity")
    return arr


def load_batch_cached(path, expected_key, n_texts):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("request_key") != expected_key:
        raise ValueError(f"Invalid cache entry: {path}")

    vectors = validate_vectors(record["vectors"], n_texts)
    return vectors, record


def request_batch(client, texts, model):
    started = time.perf_counter()
    raw_response = client.embeddings.with_raw_response.create(
        model=model, input=texts, encoding_format="float",
    )
    response = raw_response.parse()
    seconds = time.perf_counter() - started

    # Restore input order before validation.
    items = sorted(response.data, key=lambda item: item.index)
    vectors = validate_vectors(
        [item.embedding for item in items],
        len(texts),
        indices=[item.index for item in items],
    )
    return vectors, seconds, 1 + raw_response.retries_taken


def summarize_batches(batch_stats):
    """Sum generation costs once per unique batch, including cached results."""
    return {
        "endpoint_requests": sum(batch["endpoint_requests"] for batch in batch_stats.values()),
        "embedding_seconds": sum(batch["embedding_seconds"] for batch in batch_stats.values()),
        "new_endpoint_requests": sum(batch["new_endpoint_requests"] for batch in batch_stats.values()),
    }


def get_embeddings_and_stats(texts, cache_path, batch_size=64, offline=False):
    """Read or request batches, then return their vectors and generation costs."""
    base_url, api_key, model = settings()
    cache_path = Path(cache_path)
    client = None
    batches = []
    batch_stats = {}

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        key = cache_key(base_url=base_url, model=model, texts=batch_texts)
        cache_file = cache_path / f"{key}.json"
        from_cache = cache_file.exists()

        if from_cache:
            vectors, record = load_batch_cached(cache_file, key, len(batch_texts))
        else:
            if offline:
                raise FileNotFoundError(f"Missing offline embedding batch: {cache_file}")
            if client is None:
                client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=2)

            vectors, seconds, requests = request_batch(client, batch_texts, model)
            record = {
                "request_key": key,
                "model": model,
                "base_url": base_url,
                "vectors": vectors.tolist(),
                "endpoint_requests": requests,
                "embedding_seconds": seconds,
            }
            write_cache(cache_file, record)

        batches.append(vectors)
        # Repeated cache entries contribute vectors, but no additional costs.
        if key not in batch_stats:
            batch_stats[key] = {
                "endpoint_requests": record["endpoint_requests"],
                "embedding_seconds": record["embedding_seconds"],
                "new_endpoint_requests": 0 if from_cache else record["endpoint_requests"],
            }

    info = {
        **summarize_batches(batch_stats),
        "embedding_model": model,
        "base_url": base_url,
        "batch_size": batch_size,
    }
    return np.concatenate(batches), info
