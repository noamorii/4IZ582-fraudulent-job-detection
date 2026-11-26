from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from preprocessing import (
    prepare_text, preprocess_jobs, add_text_features,
    drop_empty_jobs, prepare_splits,
)
from tfidf_evaluation import evaluate_test, plot_test_results


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    results_path = data_dir / "tfidf_test_results.joblib"

    # 1. Reuse the final results if they have already been calculated.
    if results_path.exists():
        saved = joblib.load(results_path)
        print("Loaded saved test results. Models were not evaluated again.")
    else:
        # 2. Recreate the same split used for training.
        df = pd.read_csv(data_dir / "fake_job_postings.csv")
        df = prepare_text(df)
        df = preprocess_jobs(df)
        df = add_text_features(df)
        df, dropped = drop_empty_jobs(df)
        train_df, test_df, cv_splits = prepare_splits(df)

        # 3. Load the trained models and thresholds selected on train.
        baseline = joblib.load(data_dir / "baseline_training.joblib")
        forest = joblib.load(data_dir / "random_forest_training.joblib")
        models = {
            "Dummy": (baseline["dummy_search"], 0.5),
            "TF-IDF + Logistic Regression": (
                baseline["tfidf_search"], baseline["threshold"],
            ),
            "TF-IDF + Random Forest": (
                forest["rf_search"], forest["threshold"],
            ),
        }

        # 4. Evaluate once. Do not fit models or choose thresholds on test.
        results, predictions = evaluate_test(models, test_df)
        saved = {
            "results": results,
            "predictions": predictions,
            "test_ids": test_df["job_id"].tolist(),
            "best_parameters": {
                "Logistic Regression": baseline["tfidf_search"].best_params_,
                "Random Forest": forest["rf_search"].best_params_,
            },
        }
        # Save before opening any plot windows.
        joblib.dump(saved, results_path)

    # 5. Print and export the saved numbers, then display and save the plots.
    results = saved["results"]
    predictions = saved["predictions"]
    print("\nFinal test results (fraudulent class = 1):")
    print(results.round(4).to_string(index=False))
    results.to_csv(data_dir / "tfidf_test_metrics.csv", index=False)

    for name, parameters in saved["best_parameters"].items():
        print(f"\n{name} parameters selected on train: {parameters}")

    for item in predictions:
        matrix = confusion_matrix(item["actual"], item["predicted"], labels=[0, 1])
        print(f'\n{item["name"]}: confusion matrix [[TN, FP], [FN, TP]]')
        print(matrix)

    print(f"\nMetrics and PNG plots are saved in: {data_dir}")
    plot_test_results(predictions, save_dir=data_dir)
