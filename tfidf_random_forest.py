from tempfile import TemporaryDirectory

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from tfidf_evaluation import train_with_cv


def train_tfidf_rf(train_df, cv_splits, inner_splits, final_splits):
    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__max_depth": [30, None],
        "classifier__min_samples_leaf": [1, 3],
    }
    with TemporaryDirectory(prefix="tfidf-rf-") as cache:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, min_df=2)),
            ("classifier", RandomForestClassifier(
                n_estimators=100, class_weight="balanced_subsample",
                n_jobs=2, random_state=42,
            )),
        ], memory=cache)
        result = train_with_cv(
            pipeline, train_df, cv_splits, inner_splits, final_splits, parameters,
        )
        result["model"].memory = None
    return result
