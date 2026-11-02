# 1. Import libraries and load the data
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/fake_job_postings.csv")

# 2. Class balance
# 0 = legitimate, 1 = fraudulent.
counts = df["fraudulent"].value_counts().sort_index()
percentages = counts / len(df) * 100
print(pd.DataFrame({"Count": counts, "Percent": percentages.round(2)}))

# 3. Plot class balance
plt.figure(figsize=(7, 4))
plt.bar(["Legitimate (0)", "Fraudulent (1)"], counts)
plt.title("Job Posting Class Balance")
plt.ylabel("Number of Job Postings")
plt.tight_layout()
plt.show()

# 4. Explain why accuracy is not enough
# Accuracy can be misleading when one class is much larger.
print(f"Always predicting legitimate gives {percentages[0]:.2f}% accuracy.")
print("But it detects no fraud. Also evaluate fraud precision, recall and F1.")

# 5. Replace missing text with empty strings
columns = ["title", "company_profile", "description", "requirements", "benefits"]
missing = df[columns].isna()
print("Missing values before filling:")
print(missing.sum())

df[columns] = df[columns].fillna("")

# Check that each missing value became an empty string, not nan or None text.
for column in columns:
    assert (df.loc[missing[column], column] == "").all()

# 6. Combine the text fields
# Join the five fields for each job posting, using a space between fields.
df["text"] = df[columns].agg(" ".join, axis=1)
assert df["text"].notna().all()

print(df[["title", "text"]].head())
