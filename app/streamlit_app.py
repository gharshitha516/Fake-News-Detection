
import streamlit as st
import joblib
import re
import string
import os

st.set_page_config(page_title="Fake News Detection", page_icon="🗞️")

@st.cache_resource
def load_model():
    model_path = os.path.join("..", "model", "fake_news_detection.pkl")
    vectorizer_path = os.path.join("..", "model", "tfidf_vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("🗞️ Fake News Detector")
news_input = st.text_area("Please enter your news article:", height=200)

if st.button("🔍 Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        cleaned_text = clean_text(news_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.success("🟢 This news is **REAL**.")
        else:
            st.error("🔴 This news is **FAKE**.")
