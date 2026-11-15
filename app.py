import streamlit as st
import joblib

from pathlib import Path

base = Path(__file__).resolve().parent

model = joblib.load(base / "fake_news_model.pkl")
vectorizer = joblib.load(base / "tfidf_vectorizer.pkl")

st.title("Fake News Detection System")

st.write("Enter a news article to check its prediction.")

news = st.text_area("Enter News Here:")

if st.button("Check News"):

    if news.strip():

        transformed_text = vectorizer.transform([news])

        prediction = model.predict(transformed_text)

        if prediction[0] == 0:
            st.error("Prediction: FAKE NEWS")
        else:
            st.success("Prediction: REAL NEWS")

    else:
        st.warning("Please enter some news.")