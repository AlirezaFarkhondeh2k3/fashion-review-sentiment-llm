import json
from transformers import pipeline

# Use a small model
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Load and filter reviews > 100 words
reviews = []
with open(r"C:\NLP\GroupProd\Amazon_Fashion.jsonl\Amazon_Fashion.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        text = json.loads(line).get("text", "")
        if len(text.split()) > 100:
            reviews.append(text)
        if len(reviews) == 10:
            break

# Summarize each review and print original + summary
summaries = []
for i, review in enumerate(reviews, 1):
    summary = summarizer(review, max_length=60, min_length=30, do_sample=False)[0]["summary_text"]
    summaries.append(summary)
    print(f"Review {i} Original:\n{review}\n")
    print(f"Review {i} Summary:\n{summary}\n{'-'*80}")

# Final 50-word combined summary
combined = " ".join(summaries)
final_summary = summarizer(combined, max_length=50, min_length=30, do_sample=False)[0]["summary_text"]
print("\nFinal 50-word Summary Report:\n", final_summary)
