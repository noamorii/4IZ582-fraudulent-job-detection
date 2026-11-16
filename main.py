import pandas as pd
import numpy as np

from preprocessing import (
    prepare_text,
    preprocess_jobs,
    add_text_features,
    drop_empty_jobs,
    prepare_splits,
)
from embeddings import get_embeddings

# Load and preprocess
df = pd.read_csv("data/fake_job_postings.csv")
df = prepare_text(df)
cleaned = preprocess_jobs(df)
with_features = add_text_features(cleaned)
kept, dropped = drop_empty_jobs(with_features)

# Split
train_df, test_df, cv_splits = prepare_splits(kept, n_splits=5, random_state=42)
print(f"Train: {len(train_df)} | Test: {len(test_df)} | CV folds: {len(cv_splits)}")

print(type(train_df))

# Check the cleaned data before embedding
print(train_df[["title", "text"]].head())

# Embed
X_train = get_embeddings(train_df["text"].tolist(), cache_path="data/embeddings_train.npy")
X_test = get_embeddings(test_df["text"].tolist(), cache_path="data/embeddings_test.npy")
print(X_train.shape, X_test.shape)

# Comparing the newly created arrays with the train and test dataframes to see, if the number of rows stayed the same
print(X_train.shape[0] == train_df.shape[0], X_test.shape[0] == test_df.shape[0])

# Making the .npy files a bit smaller by coverting them into float32
for name in ["embeddings_train", "embeddings_test"]:
    arr = np.load(f"data/{name}.npy")
    print(name, arr.dtype, arr.shape, f"{arr.nbytes / 1e6:.0f} MB")
    np.save(f"data/{name}.npy", arr.astype(np.float32))