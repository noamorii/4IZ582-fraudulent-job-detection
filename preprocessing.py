import re
from html import unescape

import pandas as pd
from lxml import html

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

TEXT_COLUMNS = [
    "title", "company_profile", "description", "requirements", "benefits",
]

METADATA_COLUMNS = [
    "location", "department", "salary_range", "employment_type",
    "required_experience", "required_education", "industry", "function",
]


def prepare_text(df):
    # Work on a copy and replace missing text with empty strings.
    df = df.copy()
    columns = TEXT_COLUMNS
    missing = df[columns].isna()
    print("Missing values before filling:")
    print(missing.sum())

    df[columns] = df[columns].fillna("")

    # Check that missing values became empty strings, not nan or None text.
    for column in columns:
        assert (df.loc[missing[column], column] == "").all()

    # Join the five fields for each job posting, using spaces between fields.
    df["text"] = df[columns].agg(" ".join, axis=1)
    assert df["text"].notna().all()
    return df


def clean_html_text(value):
    # Turn one value into plain text with no HTML tags or extra spaces.
    # If the value is missing, return an empty string.
    if pd.isna(value):
        return ""

    # Convert HTML codes such as &amp; into normal characters such as &.
    text = unescape(str(value))

    # If HTML tags are present, read the HTML and remove scripts and styles.
    if re.search(r"<(?:/?[a-zA-Z][^>]*|!--.*?)>", text, flags=re.S):
        root = html.fragment_fromstring(text, create_parent="div")
        for element in root.xpath(".//script | .//style"):
            element.drop_tree()
        # Keep the text inside the tags, with spaces between the pieces.
        text = " ".join(root.itertext())

    # Replace repeated spaces, tabs, and line breaks with single spaces.
    return " ".join(text.split())


def preprocess_jobs(jobs):
    cleaned = jobs.copy()

    # Clean every value in the text columns
    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].map(clean_html_text)

    # add mode todo
    return cleaned


def add_text_features(jobs):
    # Add text features - word count and filled fields count for each job posting.
    result = jobs.copy()
    text_fields = result[TEXT_COLUMNS]

    # Join the text fields into one string and count its words.
    result["text"] = text_fields.agg(" ".join, axis=1).str.strip()
    result["word_count"] = result["text"].str.split().str.len()

    # Count the filled fields in each ad. Treat missing values and fields
    # with only spaces as empty.
    metadata = result[METADATA_COLUMNS].fillna("").apply(lambda col: col.str.strip())
    result["filled_fields"] = (
            text_fields.ne("").sum(axis=1)
            + metadata.ne("").sum(axis=1)
    )
    return result


def drop_empty_jobs(jobs):
    # Separate ads with text from ads with no text.
    empty = jobs["text"].eq("")
    kept = jobs.loc[~empty].copy()

    # For ads with no text, keep only the job ID and the real or fake label.
    dropped = jobs.loc[empty, ["job_id", "fraudulent"]].copy()
    return kept, dropped


def handle_duplicates_and_conflicts(jobs, subset=None):
    # Consistently handle duplicate postings and conflicting fraud labels (0 vs 1).
    df = jobs.copy()
    if subset is None:
        excluded = {
            "job_id", "fraudulent", "text", "word_count", "filled_fields", "group_id"
        }
        subset = [col for col in df.columns if col not in excluded]

    # 1. Find duplicate groups where identical features have conflicting labels (both 0 and 1).
    label_counts = df.groupby(subset, dropna=False)["fraudulent"].transform("nunique")
    conflicting_mask = label_counts > 1
    conflicts_removed = int(conflicting_mask.sum())

    # Remove conflicting rows first so noisy labels do not corrupt training or testing.
    df_clean = df.loc[~conflicting_mask].copy()

    # 2. Drop exact duplicate postings across the feature columns, keeping the first one.
    deduplicated = df_clean.drop_duplicates(subset=subset, keep="first").copy()
    duplicates_removed = len(df_clean) - len(deduplicated)

    return deduplicated, duplicates_removed, conflicts_removed


def assign_posting_groups(jobs):
    # Assign a group_id so identical or related postings (sharing the same cleaned
    # description or full text across different cities/titles) stay in the same split.
    result = jobs.copy()
    if "text" not in result.columns:
        result["text"] = result[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).str.strip()

    # Group by normalized description; fall back to full text if description is empty.
    desc_key = result["description"].fillna("").str.strip().str.lower()
    text_key = result["text"].fillna("").str.strip().str.lower()
    group_key = desc_key.mask(desc_key.eq(""), text_key)

    result["group_id"] = pd.factorize(group_key)[0]
    return result


def verify_no_split_leakage(train_df, test_df, cv_splits):
    # Check that there is zero overlap of job_ids, related groups, or identical texts
    # between train and test sets, and across all cross-validation folds.
    assert set(train_df["job_id"]).isdisjoint(set(test_df["job_id"])), (
        "Leakage error: job_id overlap between train and test!"
    )
    assert set(train_df["group_id"]).isdisjoint(set(test_df["group_id"])), (
        "Leakage error: related postings (group_id) overlap between train and test!"
    )
    assert set(train_df["text"]).isdisjoint(set(test_df["text"])), (
        "Leakage error: identical text overlap between train and test!"
    )

    for fold_num, (tr_idx, val_idx) in enumerate(cv_splits, start=1):
        fold_train = train_df.iloc[tr_idx]
        fold_val = train_df.iloc[val_idx]

        assert set(fold_train["job_id"]).isdisjoint(set(fold_val["job_id"])), (
            f"Fold {fold_num} leakage: job_id overlap!"
        )
        assert set(fold_train["group_id"]).isdisjoint(set(fold_val["group_id"])), (
            f"Fold {fold_num} leakage: related postings (group_id) overlap!"
        )
        assert set(fold_train["text"]).isdisjoint(set(fold_val["text"])), (
            f"Fold {fold_num} leakage: identical text overlap!"
        )


def prepare_splits(jobs, n_splits=5, random_state=42):
    # 1. Remove duplicates and conflicting labels
    deduplicated, dup_removed, conf_removed = handle_duplicates_and_conflicts(jobs)
    print(f"Duplicates removed: {dup_removed} | Conflicting rows removed: {conf_removed}")

    # 2. Assign group IDs to link related job postings
    grouped = assign_posting_groups(deduplicated).reset_index(drop=True)

    # 3. Hold out ~20% for the test set (1 out of 5 stratified group folds)
    outer_cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(
        outer_cv.split(grouped, grouped["fraudulent"], grouped["group_id"])
    )

    train_df = grouped.iloc[train_idx].reset_index(drop=True)
    test_df = grouped.iloc[test_idx].reset_index(drop=True)

    # 4. Prepare identical cross-validation folds on train_df for all models
    inner_cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    cv_splits = list(
        inner_cv.split(train_df, train_df["fraudulent"], train_df["group_id"])
    )

    # 5. Verify no identical or related postings cross between any splits
    verify_no_split_leakage(train_df, test_df, cv_splits)
    print("Leakage check passed: 0 overlapping texts or related groups across splits.")

    return train_df, test_df, cv_splits
