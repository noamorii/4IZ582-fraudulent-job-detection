from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import (
    prepare_text, preprocess_jobs, add_text_features,
    drop_empty_jobs, prepare_splits,
)
from tfidf_evaluation import prepare_inner_splits, select_oof_threshold
from tfidf_logistic_regression_and_dummy import train_dummy, train_tfidf
from tfidf_random_forest import train_tfidf_rf


if __name__ == "__main__":
    # 1. Prepare data and create the shared outer and inner CV folds once.
    data_dir = Path(__file__).resolve().parent / "data"
    df = pd.read_csv(data_dir / "fake_job_postings.csv")
    df = prepare_text(df)
    df = preprocess_jobs(df)
    df = add_text_features(df)
    df, dropped = drop_empty_jobs(df)
    train_df, test_df, cv_splits = prepare_splits(df)
    inner_cv_splits, final_inner_splits = prepare_inner_splits(train_df, cv_splits)
    y_train = train_df["fraudulent"].to_numpy()

    # 2. Load saved nested-CV models, or train them if the files are missing.
    # These files belong to this fixed dataset and search configuration.

    baseline_path = data_dir / "baseline_training.joblib"
    forest_path = data_dir / "random_forest_training.joblib"
    if baseline_path.exists():
        baseline = joblib.load(baseline_path)
        dummy_result = baseline["dummy_result"]
        tfidf_lr_result = baseline["logistic_result"]
        print("Loaded Dummy and TF-IDF + Logistic Regression (nested CV).")
    else:
        dummy_result = train_dummy(train_df, cv_splits, inner_cv_splits, final_inner_splits)
        tfidf_lr_result = train_tfidf(train_df, cv_splits, inner_cv_splits, final_inner_splits)

    if forest_path.exists():
        tfidf_rf_result = joblib.load(forest_path)
        print("Loaded TF-IDF + Random Forest (nested CV).")
    else:
        tfidf_rf_result = train_tfidf_rf(train_df, cv_splits, inner_cv_splits, final_inner_splits)

    model_results = {
        "Dummy": dummy_result,
        "TF-IDF + Logistic Regression": tfidf_lr_result,
        "TF-IDF + Random Forest": tfidf_rf_result,
    }

    # 3. Select thresholds from the nested OOF predictions.
    threshold_rows = []
    for name, result in [
        ("TF-IDF + Logistic Regression", tfidf_lr_result),
        ("TF-IDF + Random Forest", tfidf_rf_result),
    ]:
        # Reject a cache created for different training rows or CV indices.
        assert result["train_ids"] == train_df["job_id"].tolist()
        for stored, current in zip(result["outer_splits"], cv_splits):
            assert all(np.array_equal(a, b) for a, b in zip(stored, current))
        for stored_fold, current_fold in zip(result["inner_splits"], inner_cv_splits):
            for stored, current in zip(stored_fold, current_fold):
                assert all(np.array_equal(a, b) for a, b in zip(stored, current))

        table = select_oof_threshold(y_train, result["oof_scores"])
        result["threshold"] = float(table.loc[0, "Threshold"])
        threshold_rows.append(table.head(1).assign(Model=name))

    joblib.dump({"dummy_result": dummy_result, "logistic_result": tfidf_lr_result}, baseline_path)
    joblib.dump(tfidf_rf_result, forest_path)
    threshold_table = pd.concat(threshold_rows, ignore_index=True)
    with pd.option_context("display.max_columns", None):
        print(threshold_table.round(4).to_string(index=False))

    # 4. Compare outer-fold scores at the default threshold.
    summary_rows = []
    for name, result in model_results.items():
        summary_rows.append({
            "Model": name,
            "Mean outer F1": np.mean(result["fold_f1"]),
            "Outer F1 standard deviation": np.std(result["fold_f1"]),
            "Mean outer AP": np.mean(result["fold_ap"]),
            "CV and refit time (s)": result["seconds"],
            "Selected parameters": result["best_params"],
        })
    model_summary = pd.DataFrame(summary_rows).set_index("Model")
    with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
        print(model_summary.round(4).to_string())

    print("\nNested-CV results loaded or trained. Test evaluation was not run.")
