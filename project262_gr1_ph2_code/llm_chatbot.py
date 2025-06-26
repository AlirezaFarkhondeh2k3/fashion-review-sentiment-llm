import pandas as pd
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Step 1: Load review with question
def load_reviews_with_questions(filename):
    try:
        df = pd.read_json(filename, lines=True)

        if df.empty:
            print("Error: The file is empty")
            return None

        text_columns = [col for col in df.columns if col.lower() in ['text', 'review', 'content', 'body', 'reviewtext']]
        if not text_columns:
            print("Error: Could not find review text column.")
            return None

        text_column = text_columns[0]
        print(f"Using column '{text_column}' for review text")

        question_reviews = df[df[text_column].str.contains(r'\?', na=False)]
        if question_reviews.empty:
            print("No question-type reviews found.")
            return None

        selected = question_reviews.sample(1).iloc[0]
        return selected[text_column]

    except Exception as e:
        print(f"Error: {e}")
        return None

# Step 2: Load T5-small
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 3: Generate a response
review = load_reviews_with_questions(r"C:\NLP\GroupProd\AMAZON_FASHION_5.json")

if review:
    print("\nSelected Review (with question):")
    print(review)

    # Format prompt as a task for T5 — T5 likes clear instruction
    prompt = f"answer question: {review}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        num_beams=5,
        early_stopping=True
    )

    reply = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\nAuto-Generated Service Representative Response:\n", reply)

else:
    print("No valid review found.")
