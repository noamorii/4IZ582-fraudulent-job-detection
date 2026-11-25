import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def tune_and_train_logistic_regression(X_train, y_train, cv_splits, random_state=42):
    base_model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_splits,
        scoring=SCORING,
        n_jobs=-1,
        refit="f1",
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid


def summarize_cv_results(search, model_name="Embeddings Logistic Regression"):
    """Extract mean CV metrics for the best model at the default 0.5 threshold."""
    best_idx = search.best_index_
    cv = search.cv_results_
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "best_params": str(search.best_params_),
                "accuracy": cv["mean_test_accuracy"][best_idx],
                "precision": cv["mean_test_precision"][best_idx],
                "recall": cv["mean_test_recall"][best_idx],
                "f1": cv["mean_test_f1"][best_idx],
            }
        ]
    )


def select_threshold_embeddings(search, X_train, y_train, cv_splits):
    """Select the probability threshold using out-of-fold training predictions only."""
    # Each row is predicted by a model trained on the other CV folds.
    probabilities = cross_val_predict(
        search.best_estimator_,
        X_train,
        y_train,
        cv=cv_splits,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_train,
        probabilities,
    )
    # The last precision/recall point has no threshold, so leave it out.
    results = pd.DataFrame(
        {
            "Threshold": thresholds,
            "Train OOF precision": precision[:-1],
            "Train OOF recall": recall[:-1],
        }
    )
    precision = results["Train OOF precision"]
    recall = results["Train OOF recall"]
    # F1 is zero when both precision and recall are zero.
    results["Train OOF F1"] = (
        2 * precision * recall / (precision + recall)
    ).fillna(0)

    # Maximize train OOF F1. On a tie, choose the higher threshold.
    results = results.sort_values(
        ["Train OOF F1", "Threshold"],
        ascending=[False, False],
    ).reset_index(drop=True)
    threshold = float(results.loc[0, "Threshold"])
    return threshold, results


def evaluate_on_test(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    print(f"\n=== Held-Out Test Evaluation (threshold={threshold:.4f}) ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, preds, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=4, zero_division=0))