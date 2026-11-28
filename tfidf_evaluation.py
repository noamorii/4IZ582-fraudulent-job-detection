from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, make_scorer,
    average_precision_score, roc_auc_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold


SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}

INNER_CV_FOLDS = 3
INNER_CV_SEED = 1
F1_SCORER = make_scorer(f1_score, zero_division=0)


def prepare_inner_splits(train_df, cv_splits):
    # Make the same inner splits as the team's embeddings experiment.
    y = train_df["fraudulent"].to_numpy()
    groups = train_df["group_id"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=INNER_CV_FOLDS, shuffle=True, random_state=INNER_CV_SEED,
    )
    inner_splits = []
    for fit, validation in cv_splits:
        folds = list(splitter.split(np.zeros(len(fit)), y[fit], groups[fit]))
        for inner_fit, inner_validation in folds:
            assert set(groups[fit[inner_fit]]).isdisjoint(groups[fit[inner_validation]])
        inner_splits.append(folds)

    final_splits = list(splitter.split(np.zeros(len(y)), y, groups))
    return inner_splits, final_splits


def fit_on_train(estimator, X, y, param_grid, inner_splits):
    # All representations receive exactly the same inner row indices.
    search = GridSearchCV(
        estimator, param_grid, scoring=F1_SCORER,
        cv=inner_splits, refit=True, n_jobs=1, error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def train_with_cv(estimator, train_df, outer_splits, inner_cv_splits,
                  final_inner_splits, param_grid):
    X = train_df["text"].to_numpy()
    y = train_df["fraudulent"].to_numpy()
    if isinstance(estimator, DummyClassifier):
        X = X.reshape(-1, 1)
    started = perf_counter()
    fold_f1 = []
    fold_ap = []
    fold_parameters = []
    oof_scores = np.full(len(y), np.nan)

    for fold, (fit, validation) in enumerate(outer_splits):
        print(f"Outer fold {fold + 1}/{len(outer_splits)}", flush=True)
        validation_model, parameters = fit_on_train(
            estimator, X[fit], y[fit], param_grid, inner_cv_splits[fold],
        )
        positive_column = list(validation_model.classes_).index(1)
        scores = validation_model.predict_proba(X[validation])[:, positive_column]
        oof_scores[validation] = scores
        fold_f1.append(F1_SCORER(validation_model, X[validation], y[validation]))
        fold_ap.append(average_precision_score(y[validation], scores))
        fold_parameters.append(parameters)

    assert np.isfinite(oof_scores).all(), "Missing out-of-fold scores"
    print("Final inner search and refit on all train", flush=True)
    final_model, best_params = fit_on_train(
        estimator, X, y, param_grid, final_inner_splits,
    )
    return {
        "model": final_model,
        "best_params": best_params,
        "fold_f1": fold_f1,
        "fold_ap": fold_ap,
        "fold_parameters": fold_parameters,
        "oof_scores": oof_scores,
        "seconds": perf_counter() - started,
        "train_ids": train_df["job_id"].tolist(),
        "outer_splits": outer_splits,
        "inner_splits": inner_cv_splits,
        "final_inner_splits": final_inner_splits,
    }


def select_oof_threshold(actual, scores):
    precision, recall, thresholds = precision_recall_curve(actual, scores)
    table = pd.DataFrame({
        "Threshold": thresholds,
        "OOF precision": precision[:-1],
        "OOF recall": recall[:-1],
    })
    precision = table["OOF precision"]
    recall = table["OOF recall"]
    table["OOF F1"] = (2 * precision * recall / (precision + recall)).fillna(0)
    return table.sort_values(
        ["OOF F1", "Threshold"], ascending=[False, False],
    ).reset_index(drop=True)


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


def plot_confusion_matrix(actual, predicted, model_name, save_path=None, show=True):
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
    ax.set_title(f"{model_name}\nTest confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return matrix


def plot_test_results(predictions, save_dir=None):
    # Draw charts from saved predictions, without fitting or evaluating models.
    for number, item in enumerate(predictions, start=1):
        save_path = None
        if save_dir is not None:
            save_path = save_dir / f"tfidf_test_confusion_{number}.png"
        plot_confusion_matrix(
            item["actual"], item["predicted"],
            f'{item["name"]} (threshold={item["threshold"]:.4f})',
            save_path=save_path, show=False,
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
    prevalence = predictions[0]["actual"].mean()
    pr_ax.axhline(prevalence, color="gray", linestyle="--", label="Fraud prevalence")
    pr_ax.set_title("Test precision–recall curves")
    pr_ax.legend()
    roc_fig.tight_layout()
    pr_fig.tight_layout()
    if save_dir is not None:
        roc_fig.savefig(save_dir / "tfidf_test_roc.png", dpi=150, bbox_inches="tight")
        pr_fig.savefig(save_dir / "tfidf_test_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.show()
