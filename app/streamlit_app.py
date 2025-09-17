import streamlit as st
import joblib
import numpy as np
import re

# ---------- Stylometric Features ----------
def stylometric_features(text):
    words = text.split()
    num_words = len(words)
    num_chars = len(text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_len = np.mean([len(s.split()) for s in re.split(r'[.!?]', text) if s]) if text else 0
    return np.array([num_words, num_chars, avg_word_len, avg_sentence_len])

# Load model + embedder
model = joblib.load("models/text_classifier.pkl")
embedder = joblib.load("models/embedder.pkl")

st.title("🤖 AI vs Human Text Classifier (Hybrid Model)")

text = st.text_area("Enter text to classify:")

if st.button("Predict"):
    # Compute embeddings + stylometric features
    emb = embedder.encode([text], convert_to_numpy=True)
    style = stylometric_features(text).reshape(1, -1)
    X = np.hstack([emb, style])

    # Predict + probability
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    st.subheader("🔎 Prediction Result")
    if pred == 0:
        st.success(f"🟢 Human-written (Confidence: {probs[0]*100:.2f}%)")
    else:
        st.error(f"🔴 AI-generated (Confidence: {probs[1]*100:.2f}%)")

    st.write("📊 Confidence Scores")
    st.write(f"🟢 Human: {probs[0]*100:.2f}%")
    st.write(f"🔴 AI: {probs[1]*100:.2f}%")
