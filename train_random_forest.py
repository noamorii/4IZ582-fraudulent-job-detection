"""Olesia: tune embeddings -> Random Forest; export shared CV/time table."""

import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from model_comparison import load_embedding_dataset, tune_and_evaluate_model

RF_PARAMETERS = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_leaf": [1, 5, 10],
    "clf__max_features": ["sqrt", 0.5, 1.0],
    "clf__class_weight": [None, "balanced", "balanced_subsample"],
}


def train_random_forest(dataset_folder, output_folder, n_iter=12, n_jobs=1):
    vectors, metadata, manifest, train_mask, cv_splits = load_embedding_dataset(dataset_folder)
    X_train = vectors[train_mask]
    y_train = metadata.loc[train_mask, "fraudulent"].to_numpy()
    print(f"Source: {manifest['source']}; train={len(y_train)}, folds={len(cv_splits)}", flush=True)
    # Parallelism only at search level, avoiding nested worker pools.
    search, table = tune_and_evaluate_model(
        "embeddings_random_forest",
        Pipeline([("clf", RandomForestClassifier(random_state=42, n_jobs=1))]),
        RF_PARAMETERS, X_train, y_train, cv_splits, manifest, output_folder,
        n_iter=n_iter, n_jobs=n_jobs,
        groups=metadata.loc[train_mask, "group_id"].to_numpy(),
    )
    print(f"Best RF params: {search.best_params_}", flush=True)
    columns = ["model", "is_mock", "cv_average_precision_mean", "cv_average_precision_std",
               "nested_cv_seconds", "search_seconds", "total_model_seconds"]
    current = table["experiment"].eq(manifest["metadata_sha256"][:16]) & table["source"].eq(manifest["source"])
    print(table.loc[current, columns].to_string(index=False))
    return search, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/school_embeddings"))
    parser.add_argument("--output", type=Path, default=Path("outputs/real_models"))
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    train_random_forest(args.cache, args.output, args.n_iter, args.n_jobs)
