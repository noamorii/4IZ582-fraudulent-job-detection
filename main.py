"""Prepare shared splits and school embeddings; importing makes no API calls."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from embedding_export import save_cache, metadata_signature
from embeddings import get_embeddings_and_stats
from preprocessing import preprocess_jobs, add_text_features, drop_empty_jobs, prepare_splits


def prepare_data(csv_path="data/fake_job_postings.csv", n_splits=5):
    jobs = pd.read_csv(csv_path)
    if jobs.job_id.isna().any() or jobs.job_id.duplicated().any():
        raise ValueError("job_id must be unique and non-missing")
    if jobs.fraudulent.isna().any() or set(jobs.fraudulent.unique()) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1")
    cleaned = add_text_features(preprocess_jobs(jobs))
    kept, dropped = drop_empty_jobs(cleaned)
    print(f"Empty postings removed: {len(dropped)}")
    train_df, test_df, cv = prepare_splits(kept, n_splits=n_splits, random_state=42)
    train_df, test_df = train_df.copy(), test_df.copy()
    train_df["split"], test_df["split"] = "train", "test"
    train_df["cv_fold"], test_df["cv_fold"] = -1, -1
    for fold, (_, valid) in enumerate(cv):
        train_df.loc[valid, "cv_fold"] = fold
    metadata = pd.concat([train_df, test_df], ignore_index=True)
    metadata["text_hash"] = metadata.text.map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    columns = ["job_id", "fraudulent", "text", "text_hash", "group_id", "split", "cv_fold"]
    return metadata[columns], cv


def build_cache(csv_path="data/fake_job_postings.csv", output_dir="data/school_embeddings",
                batch_size=64, offline=True):
    metadata, cv = prepare_data(csv_path)
    output = Path(output_dir)
    vectors, info = get_embeddings_and_stats(
        metadata.text.tolist(), output / "requests", batch_size=batch_size,
        offline=offline,
    )
    manifest = save_cache(
        output, vectors, metadata, source=f"school/{info['embedding_model']}", is_mock=False,
        embedding_seconds=info["embedding_seconds"], endpoint_requests=info["endpoint_requests"],
        extra={k: v for k, v in info.items() if k not in {"embedding_seconds", "endpoint_requests"}},
    )
    print(f"Saved {vectors.shape} embeddings; new HTTP attempts: {info['new_endpoint_requests']}")
    return vectors, metadata, manifest, cv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/fake_job_postings.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/school_embeddings"))
    parser.add_argument("--batch-size", type=int, default=64, help="Number of texts per request")
    parser.add_argument("--generate", action="store_true", help="Allow school API calls; default is offline")
    args = parser.parse_args()
    # A finished CSV cache is sufficient offline; JSON batch files need not be shipped.
    if not args.generate and (args.output / "manifest.json").exists():
        from model_comparison import load_cache
        vectors, metadata, manifest, _, cv = load_cache(args.output)
        expected, _ = prepare_data(args.csv)
        if metadata_signature(metadata) != metadata_signature(expected):
            raise ValueError("Saved metadata differs from the current data/splits; regenerate the cache")
        print(f"Offline: {vectors.shape}; source={manifest['source']}; CV folds={len(cv)}")
    else:
        build_cache(args.csv, args.output, args.batch_size, offline=not args.generate)


if __name__ == "__main__":
    main()
