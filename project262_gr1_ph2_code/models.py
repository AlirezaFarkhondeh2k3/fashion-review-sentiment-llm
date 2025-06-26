# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 18:58:15 2025

@author: alire
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

# Load preprocessed data
X_train = pd.read_csv("processed_data/X_train_cleaned_ml.csv")
X_test = pd.read_csv("processed_data/X_test_cleaned_ml.csv")
y_train = pd.read_csv("processed_data/y_train.csv").values.ravel()
y_test = pd.read_csv("processed_data/y_test.csv").values.ravel()

# Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# SVM Model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
    print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
    print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Evaluate Both Models
evaluate_model(y_test, log_preds, "Logistic Regression")
evaluate_model(y_test, svm_preds, "SVM (Linear Kernel)")

# Save Trained Models
joblib.dump(log_model, "logistic_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")



# # Evaluate on training data
# train_preds_log = log_model.predict(X_train)
# evaluate_model(y_train, train_preds_log, "Logistic Regression (Train)")

# train_preds_svm = svm_model.predict(X_train)
# evaluate_model(y_train, train_preds_svm, "SVM (Train)")


# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define function for plotting confusion matrix
# def plot_confusion_matrix(y_true, y_pred, model_name, dataset_type):
#     cm = confusion_matrix(y_true, y_pred, labels=["Negative", "Neutral", "Positive"])
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
#     plt.title(f"Confusion Matrix - {model_name} ({dataset_type})")
#     plt.xlabel("Predicted Sentiment")
#     plt.ylabel("Actual Sentiment")
#     plt.tight_layout()
#     plt.show()

# # Plot for test set
# plot_confusion_matrix(y_test, log_preds, "Logistic Regression", "Test")
# plot_confusion_matrix(y_test, svm_preds, "SVM", "Test")

# # Plot for training set
# plot_confusion_matrix(y_train, train_preds_log, "Logistic Regression", "Train")
# plot_confusion_matrix(y_train, train_preds_svm, "SVM", "Train")
