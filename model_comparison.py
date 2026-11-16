"""Shared train-only CV and result export for all team members."""

import json
import hashlib
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import (
    GridSearchCV, PredefinedSplit, RandomizedSearchCV, StratifiedGroupKFold,
)

from embedding_export import metadata_signature

SCORING = {
    "average_precision": "average_precision", "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
}


def load_cache(cache_dir):
    cache = Path(cache_dir)
    metadata = pd.read_csv(cache / "metadata.csv", keep_default_na=False)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    required = {"job_id", "fraudulent", "text", "text_hash", "group_id", "split", "cv_fold"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing metadata columns: {required - set(metadata.columns)}")
    csv_path = cache / "embeddings.csv"
    cached = pd.read_csv(csv_path, keep_default_na=False)
    if set(cached.columns) != {"job_id", "text_hash", "embedding"} or cached["job_id"].duplicated().any():
        raise ValueError("Embedding CSV must have unique job_id, text_hash and embedding columns")
    if set(cached["job_id"]) != set(metadata["job_id"]):
        raise ValueError("Embedding IDs do not match metadata")
    cached = cached.set_index("job_id").loc[metadata["job_id"]]
    if cached["text_hash"].tolist() != metadata["text_hash"].tolist():
        raise ValueError("Embeddings do not match metadata texts")
    vectors = np.asarray([json.loads(value) for value in cached["embedding"]], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape != (len(metadata), manifest["dimensions"]):
        raise ValueError("Embedding shape does not match metadata/manifest")
    if not np.issubdtype(vectors.dtype, np.number) or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite numbers")
    if len(metadata) != manifest["n_rows"] or metadata_signature(metadata) != manifest["metadata_sha256"]:
        raise ValueError("Metadata checksum mismatch: row order, labels or splits changed")
    if not isinstance(manifest.get("is_mock"), bool) or not manifest.get("source"):
        raise ValueError("Manifest must explicitly identify source and is_mock")
    if (not isinstance(manifest.get("endpoint_requests"), int)
            or manifest["endpoint_requests"] < 0
            or not isinstance(manifest.get("embedding_seconds"), (int, float))
            or not np.isfinite(manifest["embedding_seconds"])
            or manifest["embedding_seconds"] < 0):
        raise ValueError("Manifest must record request count and embedding time")
    if metadata["job_id"].eq("").any() or metadata["job_id"].duplicated().any():
        raise ValueError("Missing or duplicate job IDs")
    hashes = metadata["text"].map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    if not hashes.equals(metadata["text_hash"]):
        raise ValueError("Text hashes do not match cleaned texts")
    # Duplicate texts may remain INSIDE a group, but never cross its boundary.
    if metadata.groupby("text_hash")["group_id"].nunique().gt(1).any():
        raise ValueError("Identical texts assigned to different groups")
    if metadata.groupby("group_id")["split"].nunique().gt(1).any():
        raise ValueError("Related groups overlap train/test")
    if set(metadata["split"]) != {"train", "test"}:
        raise ValueError("Expected train and test splits")
    if set(metadata["fraudulent"]) != {0, 1}:
        raise ValueError("Expected labels 0 and 1")
    train = metadata["split"].eq("train").to_numpy()
    fold_ids = metadata.loc[train, "cv_fold"].to_numpy()
    if set(fold_ids) != set(range(manifest["n_folds"])) or manifest["n_folds"] < 2:
        raise ValueError("Invalid training fold IDs")
    if not metadata.loc[~train, "cv_fold"].eq(-1).all():
        raise ValueError("Test rows must have cv_fold=-1")
    y_train = metadata.loc[train, "fraudulent"].to_numpy()
    groups = metadata.loc[train, "group_id"].to_numpy()
    if metadata.loc[train].groupby("group_id")["cv_fold"].nunique().gt(1).any():
        raise ValueError("Related groups overlap CV folds")
    cv = list(PredefinedSplit(fold_ids).split())
    for fit, valid in cv:
        if len(np.unique(y_train[fit])) != 2 or len(np.unique(y_train[valid])) != 2:
            raise ValueError("Every training/validation fold must contain both classes")
    return vectors, metadata, manifest, train, cv


def grouped_cv(y, groups, n_splits=3, seed=1):
    """Seminar 2 §4.5: respect related entities in every validation loop."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(y)), y, groups))
    for fit, valid in splits:
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError("Group leakage in cross-validation")
        if len(np.unique(y[fit])) != 2 or len(np.unique(y[valid])) != 2:
            raise ValueError("Too few independent groups/classes for requested CV")
    return splits


def run_search(model_name, estimator, parameters, X_train, y_train, cv, manifest,
               output_dir, n_iter=None, n_jobs=1, *, groups):
    """Reusable for RF, Dummy, TF-IDF Pipeline and embeddings LR.

    Caller must pass training rows in the exact metadata order and shared cv.
    Outer CV estimates the selection procedure. Inner stratified CV selects
    parameters. A separate final search refits the model on all training data.
    """
    y_train, groups = np.asarray(y_train), np.asarray(groups)
    if len(groups) != len(y_train):
        raise ValueError("One group_id is required per training row")
    inner = grouped_cv(y_train, groups)
    common = dict(estimator=estimator, scoring=SCORING, refit="average_precision",
                  cv=inner, n_jobs=n_jobs, error_score="raise", return_train_score=False)
    if n_iter is None:
        search = GridSearchCV(param_grid=parameters, **common)
    else:
        search = RandomizedSearchCV(param_distributions=parameters, n_iter=n_iter,
                                    random_state=42, **common)
    nested_started = perf_counter()
    # Seminar 2 §6.7: score the entire search on each outer validation fold.
    # Explicit loop keeps group handling visible and avoids SDK/version routing.
    nested = {f"test_{metric}": [] for metric in SCORING}
    nested.update(fit_time=[], score_time=[])
    for fit, valid in cv:
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError("Group leakage in outer CV")
        inner_splits = grouped_cv(y_train[fit], groups[fit])
        fold_search = clone(search).set_params(cv=inner_splits)
        X_fit = X_train.iloc[fit] if hasattr(X_train, "iloc") else X_train[fit]
        X_valid = X_train.iloc[valid] if hasattr(X_train, "iloc") else X_train[valid]
        fold_started = perf_counter()
        fold_search.fit(X_fit, y_train[fit])
        nested["fit_time"].append(perf_counter() - fold_started)
        score_started = perf_counter()
        for metric, scorer in SCORING.items():
            scorer = get_scorer(scorer) if isinstance(scorer, str) else scorer
            nested[f"test_{metric}"].append(scorer(fold_search, X_valid, y_train[valid]))
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
        **{metric: nested[f"test_{metric}"] for metric in SCORING},
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
        "selection_metric": "average_precision",
        "best_params": json.dumps(search.best_params_, sort_keys=True),
        "search_seconds": elapsed, "refit_seconds": search.refit_time_,
        "nested_cv_seconds": nested_seconds,
        "total_model_seconds": nested_seconds + elapsed,
        "mean_fit_seconds": search.cv_results_["mean_fit_time"][best],
        "std_fit_seconds": search.cv_results_["std_fit_time"][best],
        "mean_score_seconds": search.cv_results_["mean_score_time"][best],
        "embedding_seconds": manifest.get("embedding_seconds"),
        "endpoint_requests": manifest["endpoint_requests"],
    }
    for metric in SCORING:
        for stat in ("mean", "std"):
            values = nested[f"test_{metric}"]
            row[f"cv_{metric}_{stat}"] = float(np.mean(values) if stat == "mean" else np.std(values))
            row[f"selection_cv_{metric}_{stat}"] = search.cv_results_[f"{stat}_test_{metric}"][best]
    table_path = output / "comparison.csv"
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    if not table.empty:
        table = table.loc[~((table["experiment"] == experiment) & (table["model"] == model_name)
                           & (table["source"] == manifest["source"]))]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(table_path, index=False)
    return search, table
