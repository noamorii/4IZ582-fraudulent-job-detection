from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from embeddings import get_embeddings
from embeddings_logistic_regression import (
    select_threshold_embeddings,
    summarize_cv_results,
    tune_and_train_logistic_regression,
)
from preprocessing import (
    add_text_features,
    drop_empty_jobs,
    prepare_splits,
    prepare_text,
    preprocess_jobs,
)

# False: load saved model if the file exists.
# True: train the model again and replace the saved file.
# Set True after changing the data, preprocessing, splits, embeddings, or model settings.
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
    y_train = train_df["fraudulent"]

    model_path = project_dir / "data/embeddings_logistic_training.joblib"

    # 3. Load the saved model, or train and save it.
    if model_path.exists() and not RETRAIN:
        saved = joblib.load(model_path)
        embeddings_search = saved["embeddings_search"]
        embeddings_threshold = saved["threshold"]
        embeddings_threshold_results = saved["threshold_results"]
        print("Loaded saved model. Set RETRAIN = True if the experiment changed.")
    else:
        train_emb_path = project_dir / "data/embeddings_train.npy"
        X_train = get_embeddings(
            train_df["text"].tolist(),
            cache_path=str(train_emb_path),
        ).astype(np.float32)

        assert X_train.shape[0] == len(train_df), "Row mismatch between X_train and train_df"

        _, embeddings_search = tune_and_train_logistic_regression(
            X_train,
            y_train,
            cv_splits=cv_splits,
            random_state=42,
        )

        # Select the fraud threshold using training OOF predictions only.
        embeddings_threshold, embeddings_threshold_results = select_threshold_embeddings(
            embeddings_search,
            X_train,
            y_train,
            cv_splits,
        )

        joblib.dump(
            {
                "embeddings_search": embeddings_search,
                "threshold": embeddings_threshold,
                "threshold_results": embeddings_threshold_results,
            },
            model_path,
        )

    # 4. Show the CV results at the default threshold (0.5).
    results = summarize_cv_results(embeddings_search)
    print("\nTraining CV results at the default threshold:")
    print(results.round(4).to_string(index=False))

    # 5. Show the best threshold and its training OOF scores.
    print("\nEmbeddings Logistic Regression: selected threshold and training OOF scores")
    print(embeddings_threshold_results.head(1).round(4).to_string(index=False))

    print("\nThese are selection scores, not final test estimates.")
    print("Test evaluation was not run.")