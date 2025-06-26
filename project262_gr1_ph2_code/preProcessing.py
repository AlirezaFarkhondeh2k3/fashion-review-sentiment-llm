# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 18:51:55 2025

@author: alire
"""


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os

# Load JSONL file
file_path = r"D:\College\6th\NLP and recommendation\group project\phase2\data\Amazon_Fashion.jsonl\Amazon_Fashion.jsonl"
df = pd.read_json(file_path, lines=True)

# Drop duplicates based on hashable columns
hashable_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (list, dict))).sum() == 0]
df = df.drop_duplicates(subset=hashable_cols)

# Remove rows with missing text or rating
df = df.dropna(subset=["text", "rating"])

# Clean text function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    return ""

# Apply text cleaning
df["cleaned_text"] = df["text"].apply(clean_text)

# Label sentiment based on rating
def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["rating"].apply(label_sentiment)

# Sample 2000 reviews for ML training
df_sampled = (
    df.groupby("sentiment", group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 1000), random_state=42))
    .reset_index(drop=True)
)

# TF-IDF vectorization (max_features for efficiency and dimensionality control)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df_sampled["cleaned_text"])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Add sentiment and rating to final dataset
df_model = pd.concat([df_sampled[["rating", "sentiment"]], tfidf_df], axis=1)

# Display class distribution
print("\nClass Distribution:")
print(df_model["sentiment"].value_counts())

# Stratified train-test split
X = tfidf_df
y = df_sampled["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Save processed outputs for modeling phase
output_folder = "processed_data"
os.makedirs(output_folder, exist_ok=True)

df_model.to_csv("phase2_processed_data.csv", index=False)
X_train.to_csv(f"{output_folder}/X_train_cleaned_ml.csv", index=False)
X_test.to_csv(f"{output_folder}/X_test_cleaned_ml.csv", index=False)
y_train.to_csv(f"{output_folder}/y_train.csv", index=False)
y_test.to_csv(f"{output_folder}/y_test.csv", index=False)

print("\nFiles saved for ML training.")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


