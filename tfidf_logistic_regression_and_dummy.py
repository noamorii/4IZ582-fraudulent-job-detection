from tempfile import TemporaryDirectory

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tfidf_evaluation import train_with_cv


def train_dummy(train_df, cv_splits, inner_splits, final_splits):
    return train_with_cv(
        DummyClassifier(strategy="most_frequent"), train_df, cv_splits,
        inner_splits, final_splits, {},
    )


def train_tfidf(train_df, cv_splits, inner_splits, final_splits):
    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "classifier__C": [1, 10],
        "classifier__class_weight": [None, "balanced"],
    }
    # Cache identical transformations within the searches, without using test.
    with TemporaryDirectory(prefix="tfidf-lr-") as cache:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000)),
            ("classifier", LogisticRegression(
                solver="liblinear", max_iter=1000, random_state=42,
            )),
        ], memory=cache)
        result = train_with_cv(
            pipeline, train_df, cv_splits, inner_splits, final_splits, parameters,
        )
        result["model"].memory = None
    return result
