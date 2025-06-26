### Sentiment Analysis & Recommendation System using Amazon Fashion Reviews

This project combines traditional ML and LLMs to analyze product reviews and generate intelligent recommendations and responses.

#### 🔧 Features

* **Lexicon-based Models:** VADER, TextBlob for sentiment scoring
* **ML Models:** Logistic Regression, SVM for sentiment classification (\~65% test accuracy)
* **Recommender System:** KNN using emotion-weighted ratings via NRCLex
* **LLM Integration:** T5-based modules for review summarization and chatbot replies

#### 📁 Project Structure

```
project262_gr1_ph2_code/
│
├── llm_chatbot.py           # T5-based chatbot for mock customer support
├── llm_summery.py           # Review summarization using T5
├── models.py                # Sentiment classification models (LogReg, SVM)
├── preProcessing.py         # TF-IDF, emotion tagging with NRCLex
└── recommender_system.py    # KNN-based recommendation logic
```

#### 📄 Other Files

* `project262_gr1_report.pdf`: Final report detailing methodology and results
* `README.md`: This file
