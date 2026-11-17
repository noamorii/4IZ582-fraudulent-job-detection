"""Prepare shared splits and school embeddings; importing makes no API calls."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from embeddings import get_embeddings, metadata_hash, save_embedding_dataset
from preprocessing import preprocess_jobs, add_text_features, drop_empty_jobs, prepare_splits


def prepare_metadata(csv_path="data/fake_job_postings.csv", n_splits=5):
    """Clean postings and return their metadata with shared train/test/CV assignments."""
    jobs = pd.read_csv(csv_path)
    if jobs.job_id.isna().any() or jobs.job_id.duplicated().any():
        raise ValueError("job_id must be unique and non-missing")
    if jobs.fraudulent.isna().any() or set(jobs.fraudulent.unique()) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1")
    cleaned = add_text_features(preprocess_jobs(jobs))
    kept, dropped = drop_empty_jobs(cleaned)
    print(f"Empty postings removed: {len(dropped)}")
    train_df, test_df, cv_splits = prepare_splits(kept, n_splits=n_splits, random_state=42)
    train_df, test_df = train_df.copy(), test_df.copy()
    train_df["split"], test_df["split"] = "train", "test"
    train_df["cv_fold"], test_df["cv_fold"] = -1, -1
    for fold, (_, validation_indices) in enumerate(cv_splits):
        train_df.loc[validation_indices, "cv_fold"] = fold
    metadata = pd.concat([train_df, test_df], ignore_index=True)
    metadata["text_hash"] = metadata.text.map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    columns = ["job_id", "fraudulent", "text", "text_hash", "group_id", "split", "cv_fold"]
    return metadata[columns], cv_splits


def create_embedding_dataset(
    csv_path="data/fake_job_postings.csv",
    output_folder="data/school_embeddings",
    batch_size=64,
    offline=True,
):
    """Prepare postings, obtain embeddings and save the complete training dataset."""
    metadata, cv_splits = prepare_metadata(csv_path)
    output_folder = Path(output_folder)
    vectors, stats = get_embeddings(
        metadata.text.tolist(), output_folder / "requests", batch_size=batch_size,
        offline=offline,
    )
    manifest = save_embedding_dataset(vectors, metadata, stats, output_folder)
    print(f"Saved {vectors.shape} embeddings; new HTTP attempts: {stats['new_endpoint_requests']}")
    return vectors, metadata, manifest, cv_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/fake_job_postings.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/school_embeddings"))
    parser.add_argument("--batch-size", type=int, default=64, help="Number of texts per request")
    parser.add_argument("--generate", action="store_true", help="Allow school API calls; default is offline")
    args = parser.parse_args()
    # A finished CSV cache is sufficient offline; JSON batch files need not be shipped.
    if not args.generate and (args.output / "manifest.json").exists():
        from model_comparison import load_embedding_dataset
        vectors, metadata, manifest, _, cv_splits = load_embedding_dataset(args.output)
        expected_metadata, _ = prepare_metadata(args.csv)
        if metadata_hash(metadata) != metadata_hash(expected_metadata):
            raise ValueError("Saved metadata differs from the current data/splits; regenerate the cache")
        print(f"Offline: {vectors.shape}; source={manifest['source']}; CV folds={len(cv_splits)}")
    else:
        create_embedding_dataset(args.csv, args.output, args.batch_size, offline=not args.generate)


if __name__ == "__main__":
    main()
