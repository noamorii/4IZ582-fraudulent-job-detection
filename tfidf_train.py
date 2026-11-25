from pathlib import Path

import joblib
import pandas as pd

from preprocessing import (
    prepare_text, preprocess_jobs, add_text_features,
    drop_empty_jobs, prepare_splits,
)
from tfidf_logistic_regression_and_dummy import train_dummy, train_tfidf
from tfidf_random_forest import train_tfidf_rf
from tfidf_evaluation import compare_cv_results, select_threshold


# False: load saved models if both files exist.
# True: train all models again and replace the saved files.
# Set True after changing the data, preprocessing, splits or model settings.
RETRAIN = False


if __name__ == "__main__":
    # 1. Read and clean the data.
    project_dir = Path(__file__).resolve().parent
    df = pd.read_csv(project_dir / "data/fake_job_postings.csv")
    df = prepare_text(df)
    df = preprocess_jobs(df)
    df = add_text_features(df)
    df, dropped = drop_empty_jobs(df)

    # 2. Reserve the test set. Use the same training CV folds for all models.
    train_df, test_df, cv_splits = prepare_splits(df)

    baseline_path = project_dir / "data/baseline_training.joblib"
    forest_path = project_dir / "data/random_forest_training.joblib"

    # 3. Load all models, or train all of them together.
    if baseline_path.exists() and forest_path.exists() and not RETRAIN:
        baseline = joblib.load(baseline_path)
        forest = joblib.load(forest_path)

        dummy_search = baseline["dummy_search"]
        logistic_search = baseline["tfidf_search"]
        forest_search = forest["rf_search"]
        logistic_threshold = baseline["threshold"]
        forest_threshold = forest["threshold"]
        logistic_threshold_results = baseline["threshold_results"]
        forest_threshold_results = forest["threshold_results"]
        print("Loaded saved models. Set RETRAIN = True if the experiment changed.")
    else:
        dummy_search = train_dummy(train_df, cv_splits)
        logistic_search = train_tfidf(train_df, cv_splits)
        forest_search = train_tfidf_rf(train_df, cv_splits)

        # Select the fraud thresholds using training predictions only.
        logistic_threshold, logistic_threshold_results = select_threshold(
            logistic_search, train_df, cv_splits,
        )
        forest_threshold, forest_threshold_results = select_threshold(
            forest_search, train_df, cv_splits,
        )

        # Save models and thresholds for later use.
        joblib.dump({
            "dummy_search": dummy_search,
            "tfidf_search": logistic_search,
            "threshold": logistic_threshold,
            "threshold_results": logistic_threshold_results,
        }, baseline_path)
        joblib.dump({
            "rf_search": forest_search,
            "threshold": forest_threshold,
            "threshold_results": forest_threshold_results,
        }, forest_path)

    # 4. Show the CV results at the default threshold (0.5).
    results = compare_cv_results(dummy_search, logistic_search, forest_search)
    print("\nTraining CV results at the default threshold:")
    print(results.round(4).to_string(index=False))

    # 5. Show the best threshold and its training OOF scores for each model.
    print("\nLogistic Regression: selected threshold and training OOF scores")
    print(logistic_threshold_results.head(1).round(4).to_string(index=False))
    print("\nRandom Forest: selected threshold and training OOF scores")
    print(forest_threshold_results.head(1).round(4).to_string(index=False))

    print("\nThese are selection scores, not final test estimates.")
    print("Test evaluation was not run.")
