# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:39:55 2025

@author: alire
"""

# -*- coding: utf-8 -*-
"""
Updated for in-memory processing and sample display
"""

import pandas as pd
import re
from nrclex import NRCLex
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import random

# Load and clean the raw dataset
file_path = r"D:\College\6th\NLP and recommendation\group project\phase2\data\Amazon_Fashion.jsonl\Amazon_Fashion.jsonl"
df = pd.read_json(file_path, lines=True)
df = df[["user_id", "asin", "text", "rating"]].dropna().drop_duplicates(subset=["user_id", "asin", "text"])

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.strip()
    return ""

df["cleaned_text"] = df["text"].apply(clean_text)

print("Sample after cleaning:")
print(df[["user_id", "asin", "rating", "cleaned_text"]].head(10))


# ---------- Emotion Score Calculation ----------
def get_emotion_score(text):
    try:
        emotion_obj = NRCLex(text)
        emotion_freq = emotion_obj.raw_emotion_scores
        total = sum(emotion_freq.values())
        if total == 0:
            return 0
        pos = sum(emotion_freq.get(em, 0) for em in ['joy', 'trust'])
        neg = sum(emotion_freq.get(em, 0) for em in ['anger', 'disgust', 'fear', 'sadness'])
        return round((pos - neg) / total, 3)
    except:
        return 0

tqdm.pandas()
df["emotion_score"] = df["cleaned_text"].progress_apply(get_emotion_score)

# Apply adjustment without saving to CSV
df["adjusted_rating"] = df["rating"] * (1 + df["emotion_score"])
df["adjusted_rating"] = df["adjusted_rating"].clip(lower=1.0, upper=5.0)

# ---------- Show Before/After ----------
print("\nSample Ratings Before vs After Emotion Adjustment:")
for i, row in df.head(10).iterrows():
    print(f"Review: {row['text'][:120]}...")
    print(f"Original Rating: {row['rating']} | Emotion Score: {row['emotion_score']} | Adjusted Rating: {round(row['adjusted_rating'], 2)}")
    print("-" * 100)

# ---------- KNN Recommender ----------
# Sample 10,000 users max
df_sample = df.groupby("user_id", group_keys=False).apply(lambda x: x.sample(1, random_state=42))
if len(df_sample["user_id"].unique()) > 10000:
    users_to_keep = df_sample["user_id"].unique()[:10000]
    df = df[df["user_id"].isin(users_to_keep)]

# Pivot to user-item matrix
user_item_matrix = df.pivot_table(index="user_id", columns="asin", values="adjusted_rating").fillna(0)
sparse_matrix = csr_matrix(user_item_matrix.values)

# Fit KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(sparse_matrix)

# Recommendation function
def recommend_items(user_index, n_recommendations=5):
    distances, indices = knn.kneighbors(sparse_matrix[user_index], n_neighbors=6)
    user_id = user_item_matrix.index[user_index]
    print(f"\nRecommendations for User: {user_id}")

    sim_users = indices.flatten()[1:]  # Exclude self
    all_ratings = user_item_matrix.iloc[sim_users]
    mean_scores = all_ratings.mean().sort_values(ascending=False)

    rated_items = user_item_matrix.iloc[user_index]
    already_rated = rated_items[rated_items > 0].index
    recommendations = mean_scores.drop(labels=already_rated, errors='ignore')[:n_recommendations]

    for asin, score in recommendations.items():
        print(f"→ Item: {asin} | Predicted Rating: {round(score, 2)}")

# Test with a random user
random_user_index = random.randint(0, user_item_matrix.shape[0] - 1)
recommend_items(random_user_index)
