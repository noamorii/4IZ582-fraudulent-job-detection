"""Train-only parameter tuning with Average Precision and timing."""

import json
import hashlib
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV, PredefinedSplit, RandomizedSearchCV, StratifiedGroupKFold,
)

from embeddings import metadata_hash, validate_vectors
from preprocessing import verify_no_split_leakage

SELECTION_METRIC = "average_precision"


def load_embedding_dataset(dataset_folder):
    """Load vectors, align them with postings and verify the shared splits."""
    dataset_folder = Path(dataset_folder)
    metadata = pd.read_csv(dataset_folder / "metadata.csv", keep_default_na=False)
    embedding_table = pd.read_csv(dataset_folder / "embeddings.csv", keep_default_na=False)
    manifest = json.loads((dataset_folder / "manifest.json").read_text(encoding="utf-8"))

    if (not metadata["job_id"].is_unique or not embedding_table["job_id"].is_unique
            or set(embedding_table["job_id"]) != set(metadata["job_id"])):
        raise ValueError("Embeddings must match postings one-to-one by job_id")
    embedding_table = embedding_table.set_index("job_id").loc[metadata["job_id"]]

    text_hashes = metadata["text"].map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    if (metadata["text_hash"].tolist() != text_hashes.tolist()
            or embedding_table["text_hash"].tolist() != text_hashes.tolist()):
        raise ValueError("Embeddings do not match posting texts")
    if metadata_hash(metadata) != manifest["metadata_sha256"]:
        raise ValueError("Saved rows or split assignments have changed")

    vectors = validate_vectors(
        [json.loads(value) for value in embedding_table["embedding"]], len(metadata),
    )
    if vectors.shape[1] != manifest["dimensions"]:
        raise ValueError("Embedding dimension does not match the saved dataset")
    train_mask = metadata["split"].eq("train").to_numpy()
    train_metadata = metadata.loc[train_mask]
    test_metadata = metadata.loc[~train_mask]
    cv_splits = list(PredefinedSplit(train_metadata["cv_fold"].to_numpy()).split())
    verify_no_split_leakage(train_metadata, test_metadata, cv_splits)
    return vectors, metadata, manifest, train_mask, cv_splits


def make_grouped_cv_splits(y, groups, n_splits=3, seed=1):
    """Seminar 2 §4.5: respect related entities in every validation loop."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def tune_and_evaluate_model(model_name, estimator, parameters, X_train, y_train, cv, manifest,
               output_dir, n_iter=None, n_jobs=1, *, groups):
    """Select parameters and report train-only Average Precision and runtime.

    Caller must pass training rows in the exact metadata order and shared cv.
    Outer CV estimates the selection procedure. Inner stratified CV selects
    parameters. A separate final search refits the model on all training data.
    """
    y_train, groups = np.asarray(y_train), np.asarray(groups)
    inner = make_grouped_cv_splits(y_train, groups)
    common = dict(estimator=estimator, scoring=SELECTION_METRIC, refit=True,
                  cv=inner, n_jobs=n_jobs, error_score="raise", return_train_score=False)
    if n_iter is None:
        search = GridSearchCV(param_grid=parameters, **common)
    else:
        search = RandomizedSearchCV(param_distributions=parameters, n_iter=n_iter,
                                    random_state=42, **common)
    nested_started = perf_counter()
    # Seminar 2 §6.7: score the entire search on each outer validation fold.
    # Explicit loop keeps group handling visible and avoids SDK/version routing.
    nested = {"average_precision": [], "fit_time": [], "score_time": []}
    score_average_precision = get_scorer(SELECTION_METRIC)
    for fit, valid in cv:
        inner_splits = make_grouped_cv_splits(y_train[fit], groups[fit])
        fold_search = clone(search).set_params(cv=inner_splits)
        X_fit = X_train.iloc[fit] if hasattr(X_train, "iloc") else X_train[fit]
        X_valid = X_train.iloc[valid] if hasattr(X_train, "iloc") else X_train[valid]
        fold_started = perf_counter()
        fold_search.fit(X_fit, y_train[fit])
        nested["fit_time"].append(perf_counter() - fold_started)
        score_started = perf_counter()
        nested["average_precision"].append(
            score_average_precision(fold_search, X_valid, y_train[valid])
        )
        nested["score_time"].append(perf_counter() - score_started)
    nested_seconds = perf_counter() - nested_started
    started = perf_counter()
    search.fit(X_train, y_train)
    elapsed = perf_counter() - started
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment = manifest["metadata_sha256"][:16]
    source_id = hashlib.sha256(manifest["source"].encode()).hexdigest()[:8]
    prefix = output / f"{experiment}_{model_name}_{source_id}"
    outer_results = pd.DataFrame({
        "outer_fold": np.arange(len(cv)),
        "fit_seconds": nested["fit_time"], "score_seconds": nested["score_time"],
        "average_precision": nested["average_precision"],
    })
    outer_results.to_csv(f"{prefix}_outer_cv_results.csv", index=False)
    detailed = pd.DataFrame(search.cv_results_)
    detailed["params"] = detailed["params"].map(lambda p: json.dumps(p, sort_keys=True))
    detailed.to_csv(f"{prefix}_cv_results.csv", index=False)
    joblib.dump(search.best_estimator_, f"{prefix}_best_model.joblib")
    best = search.best_index_
    row = {
        "experiment": experiment, "model": model_name,
        "source": manifest["source"], "is_mock": manifest["is_mock"],
        "n_train": len(y_train), "n_folds": len(cv),
        "inner_folds": 3, "evaluation_protocol": "nested_cv", "n_jobs": n_jobs,
        "n_candidates": len(search.cv_results_["params"]),
        "selection_metric": SELECTION_METRIC,
        "best_params": json.dumps(search.best_params_, sort_keys=True),
        "search_seconds": elapsed, "refit_seconds": search.refit_time_,
        "nested_cv_seconds": nested_seconds,
        "total_model_seconds": nested_seconds + elapsed,
        "mean_fit_seconds": search.cv_results_["mean_fit_time"][best],
        "std_fit_seconds": search.cv_results_["std_fit_time"][best],
        "mean_score_seconds": search.cv_results_["mean_score_time"][best],
        "embedding_seconds": manifest.get("embedding_seconds"),
        "endpoint_requests": manifest["endpoint_requests"],
        "cv_average_precision_mean": float(np.mean(nested["average_precision"])),
        "cv_average_precision_std": float(np.std(nested["average_precision"])),
        "selection_cv_average_precision_mean": search.cv_results_["mean_test_score"][best],
        "selection_cv_average_precision_std": search.cv_results_["std_test_score"][best],
    }
    table_path = output / "comparison.csv"
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    if not table.empty:
        table = table.loc[~((table["experiment"] == experiment) & (table["model"] == model_name)
                           & (table["source"] == manifest["source"]))]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(table_path, index=False)
    return search, table
