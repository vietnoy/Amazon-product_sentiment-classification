import streamlit as st
from src.predict import load_model, predict

# Load model and tokenizer
@st.cache_resource
def get_model():
    return load_model("saved_model/distilbert_model")

model, tokenizer = get_model()

# Streamlit UI
st.set_page_config(page_title="Sentiment Classifier", layout="centered")
st.title("🚀 Sentiment Classification App")

text_input = st.text_area("Enter your review:", height=150)

if st.button("Analyze Sentiment"):
    if text_input.strip():
        label, prob = predict(text_input, model, tokenizer)
        st.write(f"### Prediction: {'Positive' if label == 1 else 'Negative'} ({prob:.2f} confidence)")
    else:
        st.warning("Please enter some text to analyze.")