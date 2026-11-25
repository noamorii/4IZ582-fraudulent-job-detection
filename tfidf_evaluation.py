import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, make_scorer,
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay,
)


# Precision, recall and F1 refer to fraudulent postings (class 1).
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def compare_cv_results(dummy_search=None, tfidf_search=None, rf_search=None):
    # Tuning scores at the default threshold, NOT an independent final estimate.
    rows = []
    for name, search in [
        ("Dummy", dummy_search),
        ("TF-IDF + Logistic Regression", tfidf_search),
        ("TF-IDF + Random Forest", rf_search),
    ]:
        if search is None:
            continue
        # Read the scores for the parameter combination selected by GridSearchCV.
        best = search.best_index_
        scores = search.cv_results_
        rows.append({
            "Model": name,
            "CV accuracy": scores["mean_test_accuracy"][best],
            "CV precision": scores["mean_test_precision"][best],
            "CV recall": scores["mean_test_recall"][best],
            "CV f1": scores["mean_test_f1"][best],
            "CV F1 std": scores["std_test_f1"][best],
        })
    return pd.DataFrame(rows)


def select_threshold(search, train_df, cv_splits):
    # Each row is predicted by a model trained on the other CV folds.
    probabilities = cross_val_predict(
        search.best_estimator_, train_df["text"], train_df["fraudulent"],
        cv=cv_splits, method="predict_proba", n_jobs=1,
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        train_df["fraudulent"], probabilities,
    )
    # The last precision/recall point has no threshold, so leave it out.
    results = pd.DataFrame({
        "Threshold": thresholds,
        "Train OOF precision": precision[:-1],
        "Train OOF recall": recall[:-1],
    })
    precision = results["Train OOF precision"]
    recall = results["Train OOF recall"]
    # F1 is zero when both precision and recall are zero.
    results["Train OOF F1"] = (
        2 * precision * recall / (precision + recall)
    ).fillna(0)

    # Maximize train OOF F1. On a tie, choose the higher threshold.
    results = results.sort_values(
        ["Train OOF F1", "Threshold"], ascending=[False, False],
    ).reset_index(drop=True)
    threshold = float(results.loc[0, "Threshold"])
    print(results.head(1).round(4).to_string(index=False))
    return threshold, results


def evaluate_test(models, test_df):
    # Final evaluation only. Models and thresholds must already be fixed.
    rows = []
    predictions = []
    actual = test_df["fraudulent"].to_numpy()

    for name, (model, threshold) in models.items():
        if name == "Dummy":
            texts = test_df[["text"]]
        else:
            texts = test_df["text"]

        # Column 1 contains the probability of fraud.
        probabilities = model.predict_proba(texts)[:, 1]
        if name == "Dummy":
            predicted = model.predict(texts)
        else:
            predicted = (probabilities >= threshold).astype(int)

        rows.append({
            "Model": name,
            "Threshold": threshold,
            "Test accuracy": accuracy_score(actual, predicted),
            "Test precision": precision_score(actual, predicted, zero_division=0),
            "Test recall": recall_score(actual, predicted, zero_division=0),
            "Test F1": f1_score(actual, predicted, zero_division=0),
            "Test ROC AUC": roc_auc_score(actual, probabilities),
            "Test AP": average_precision_score(actual, probabilities),
        })
        predictions.append({
            "name": name, "actual": actual, "predicted": predicted,
            "probabilities": probabilities, "threshold": threshold,
        })

    # Return the numbers and the plot data separately.
    return pd.DataFrame(rows), predictions


def plot_confusion_matrix(actual, predicted, model_name):
    # Rows are actual classes; columns are predicted classes.
    # With labels=[0, 1], the cells are [[TN, FP], [FN, TP]].
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate (0)", "Fraudulent (1)"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False,
        im_kw={"vmin": 0},
    )
    ax.set_title(f"{model_name}\nTest Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    plt.show()
    return matrix


def plot_test_results(predictions):
    # Draw charts from saved predictions, without fitting or evaluating models.
    for item in predictions:
        plot_confusion_matrix(
            item["actual"], item["predicted"],
            f'{item["name"]} (threshold={item["threshold"]:.4f})',
        )
    roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
    pr_fig, pr_ax = plt.subplots(figsize=(6, 5))
    for item in predictions:
        RocCurveDisplay.from_predictions(
            item["actual"], item["probabilities"], name=item["name"], ax=roc_ax,
        )
        PrecisionRecallDisplay.from_predictions(
            item["actual"], item["probabilities"], name=item["name"], ax=pr_ax,
        )
    roc_ax.plot([0, 1], [0, 1], "k--", label="Random ranking")
    roc_ax.set_title("Test ROC curves")
    roc_ax.legend()
    pr_ax.axhline(item["actual"].mean(), color="gray", linestyle="--", label="Fraud prevalence")
    pr_ax.set_title("Test Precision–Recall curves")
    pr_ax.legend()
    roc_fig.tight_layout()
    pr_fig.tight_layout()
    plt.show()
