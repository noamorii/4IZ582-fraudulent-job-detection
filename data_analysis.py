from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import preprocess_jobs, drop_empty_jobs, add_text_features


def analyze_class_balance(df):
    # Count legitimate (0) and fraudulent (1) job postings.
    counts = df["fraudulent"].value_counts().reindex([0, 1], fill_value=0)
    percentages = counts / len(df) * 100
    summary = pd.DataFrame({"Count": counts, "Percent": percentages.round(2)})
    print(summary)

    plt.figure(figsize=(7, 4))
    plt.bar(["Legitimate (0)", "Fraudulent (1)"], counts)
    plt.title("Job Posting Class Balance")
    plt.ylabel("Number of Job Postings")
    plt.tight_layout()
    plt.show()

    # Accuracy can be misleading when one class is much larger.
    print(f"Always predicting legitimate gives {percentages[0]:.2f}% accuracy.")
    print("But it detects no fraud. Also evaluate fraud precision, recall and F1.")
    return summary


def check_duplicates(jobs):
    # Find job posting with the same details. Ignore their IDs and whether they
    # are marked as real (0) or fake (1) when checking for matches.
    columns = jobs.columns.drop(["job_id", "fraudulent"]).tolist()
    repeated = jobs.duplicated(subset=columns, keep=False)

    summary = (
        jobs.assign(repeated=repeated)
        .groupby("fraudulent")["repeated"]
        .agg(postings="size", has_duplicate="sum")
    )

    # Check whether duplicated postings are all marked the same way (real or fake).
    labels_per_group = (
        jobs.loc[repeated]
        .groupby(columns, dropna=False)["fraudulent"]
        .nunique()
    )

    duplicate_rows_to_delete = jobs.duplicated(subset=columns).sum()
    conflicting_groups = labels_per_group.gt(1).sum()
    return summary, duplicate_rows_to_delete, conflicting_groups


def fraud_bars(jobs, feature, ax, title):
    # Draw a bar for each value in the chosen column.
    # Each bar shows the percentage of real and fake postings. ax is where to draw it.
    groups = jobs[feature]
    # Give empty values a name so they also appear in the chart.
    if groups.isna().any():
        groups = groups.astype("string").fillna("Missing")

    # Count real (0) and fake (1) postings in each group
    counts = pd.crosstab(groups, jobs["fraudulent"])
    counts = counts.reindex(columns=[0, 1], fill_value=0)
    counts.columns = ["Real", "Fake"]

    # Turn the counts into percentages. Add each group's size to its label.
    totals = counts.sum(axis=1)
    shares = counts.div(totals, axis=0) * 100
    shares.index = [f"{group}\n(total={n:,})" for group, n in totals.items()]

    # Put the real and fake parts in the same bar and label the fake percentage.
    shares.plot.bar(
        stacked=True, ax=ax, color=["#3b7dd8", "#b0413e"], rot=0
    )
    ax.bar_label(
        ax.containers[1],
        labels=[f"{pct:.1f}%" for pct in shares["Fake"]],
        padding=3,
    )
    ax.set(title=title, xlabel="", ylabel="Amount (%)", ylim=(0, 110))
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc="lower right")
    # Return the original counts so the caller can also print or use them.
    return counts


def plot_job_features(jobs):
    # Show by graph how many job postings are real or fake in each group
    # Based on the company logo, remote work, and job type.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    fraud_bars(jobs, "has_company_logo", axes[0], "Company logo")
    fraud_bars(jobs, "telecommuting", axes[1], "Remote-work flag")
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(10, 4))
    fraud_bars(jobs, "employment_type", ax, "Employment type")
    fig.tight_layout()


def plot_text_length(jobs):
    # Split posting into four groups based on how many words they contain.
    # Show the real and fake percentages and return the number of posting in each group.
    groups = pd.cut(
        jobs["word_count"],
        bins=[0, 30, 100, 300, float("inf")],
        labels=["1–30", "31–100", "101–300", "301+"],
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    counts = fraud_bars(
        jobs.assign(text_length_group=groups),
        "text_length_group",
        ax,
        "Text length in words",
    )
    fig.tight_layout()
    return counts


def get_short_jobs(jobs, max_words=30):
    # Find posting with 1 to max_words words (30 by default).
    # Keep the columns listed below and return the shortest posting first.
    columns = ["job_id", "fraudulent", "word_count", "filled_fields", "text"]
    short = jobs.loc[jobs["word_count"].between(1, max_words), columns]
    return short.sort_values("word_count")


def main():
    csv_path = Path(__file__).parent / "data" / "fake_job_postings.csv"
    jobs = pd.read_csv(csv_path)

    analyze_class_balance(jobs)

    summary, duplicate_rows_to_delete, conflicting_groups = check_duplicates(jobs)
    print(summary)
    print(f"Duplicate rows to delete: {duplicate_rows_to_delete}")
    print(f"Duplicate groups with conflicting labels: {conflicting_groups}")
    plot_job_features(jobs)

    cleaned = preprocess_jobs(jobs)
    with_features = add_text_features(cleaned)
    kept, dropped = drop_empty_jobs(with_features)

    print(f"\nEmpty postings removed: {len(dropped)}")
    print(plot_text_length(kept))

    short = get_short_jobs(kept)
    print(f"\nPostings with 1–30 words: {len(short)}")
    print(short.head(10).to_string(index=False))

    plt.show()


if __name__ == "__main__":
    main()
