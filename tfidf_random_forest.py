from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from tfidf_evaluation import SCORING


def train_tfidf_rf(train_df, cv_splits):
    # Fit TF-IDF inside each fold, just as for Logistic Regression.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, min_df=2)),
        ("classifier", RandomForestClassifier(
            n_estimators=100, class_weight="balanced_subsample",
            n_jobs=2, random_state=42,
        )),
    ])
    # 8 combinations x 5 folds = 40 fits, plus the final refit on all train.
    # Keep the grid small enough for a student laptop.
    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__max_depth": [30, None],
        "classifier__min_samples_leaf": [1, 3],
    }
    search = GridSearchCV(
        pipeline, parameters, scoring=SCORING, refit="f1", cv=cv_splits,
        n_jobs=1, verbose=2, error_score="raise",
    )
    search.fit(train_df["text"], train_df["fraudulent"])
    print("Best Random Forest parameters:", search.best_params_)
    return search
