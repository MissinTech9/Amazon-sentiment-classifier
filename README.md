# Amazon-sentiment-classifier
An NLP-powered machine learning web application that classifies the emotional tone of Amazon product reviews using Naive Bayes and TF-IDF.
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-88.31%25-brightgreen.svg)

## 📌 Project Overview
This project is an end-to-end Natural Language Processing (NLP) machine learning pipeline designed to analyze customer feedback. It reads Amazon product reviews and classifies the emotional tone as either **Positive (1)** or **Negative (0)**. 

The project was built to demonstrate full-lifecycle model deployment, including data preprocessing, feature extraction, model training, and the creation of an interactive web interface for real-time inference.

## 🧠 Model Architecture & Pipeline
* **Text Vectorization:** `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) limited to the top 5,000 most frequent words to optimize memory and exclude English stop words.
* **Classification Algorithm:** `MultinomialNB` (Naive Bayes), which is highly scalable and traditionally performs exceptionally well on text classification tasks.
* **Pipeline Integration:** Packaged using Scikit-Learn's `make_pipeline` for seamless data transformation and prediction.

## 📊 Dataset & Performance
The model was trained on a dataset of **30,846 Amazon product reviews**. After rigorous data cleaning and dropping null values, the dataset was split into an 80/20 train-test split (24,676 training samples / 6,170 testing samples).

**Model Evaluation Metrics:**
* **Overall Accuracy:** 88.31%
* **Precision (Positive Class):** 0.89
* **Recall (Positive Class):** 0.98
* **F1-Score (Weighted Average):** 0.87

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/amazon-product-sentiment-analysis.git](https://github.com/yourusername/amazon-product-sentiment-analysis.git)
cd amazon-product-sentiment-analysis

 Install dependencies
Bash
pip install -r requirements.txt

# Launch the Web Application

Bash
streamlit run untitled4.py
(The application will automatically open in your default web browser at http://localhost:8501)

📂 Repository Structure
Plaintext
├── amazon_sentiment_analysis.ipynb   # Complete data exploration and model training notebook
├── app.py                            # Streamlit web application script
├── my_sentiment_model.pkl            # Serialized Scikit-Learn pipeline
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation

🛠️ Tech Stack
Data Manipulation: pandas

Machine Learning: scikit-learn

Model Serialization: joblib

Web Interface: streamlit & gradio
