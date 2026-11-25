from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from tfidf_evaluation import SCORING


def train_dummy(train_df, cv_splits):
    # Always predict the most frequent class. No text features are needed.
    # An empty grid evaluates this single baseline on the shared CV folds.
    search = GridSearchCV(
        DummyClassifier(strategy="most_frequent"),
        param_grid={}, scoring=SCORING, refit="f1", cv=cv_splits,
        error_score="raise",
    )
    search.fit(train_df[["text"]], train_df["fraudulent"])
    return search


def train_tfidf(train_df, cv_splits):
    # The pipeline learns the vocabulary only from each fold's training data.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000)),
        ("classifier", LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=42,
        )),
    ])

    # 16 combinations, each evaluated on the same 5 folds.
    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],  # Words or words + word pairs.
        "tfidf__min_df": [2, 5],  # Minimum documents containing a term.
        "classifier__C": [1, 10],  # Smaller C means stronger regularization.
        "classifier__class_weight": [None, "balanced"],
    }
    search = GridSearchCV(
        pipeline, parameters, scoring=SCORING, refit="f1", cv=cv_splits,
        n_jobs=1, verbose=2, error_score="raise",
    )
    search.fit(train_df["text"], train_df["fraudulent"])
    print("Best parameters:", search.best_params_)
    return search
