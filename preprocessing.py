import re
from html import unescape

import pandas as pd
from lxml import html

TEXT_COLUMNS = [
    "title", "company_profile", "description", "requirements", "benefits",
]

METADATA_COLUMNS = [
    "location", "department", "salary_range", "employment_type",
    "required_experience", "required_education", "industry", "function",
]


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
