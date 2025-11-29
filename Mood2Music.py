import streamlit as st
import pandas as pd
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# Load Models (safe + lightweight)
# -------------------------------
@st.cache_resource
def load_text_classifier():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_music_data():
    df = pd.read_csv("songs.csv")
    return df

# -------------------------------
# Text-based Emotion Detection
# -------------------------------
def predict_text_emotion(text, model):
    if not text.strip():
        return "neutral"
    preds = model(text)[0]
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    return preds[0]["label"]

# -------------------------------
# Recommendation Logic
# -------------------------------
def recommend_songs(df, mood):
    mood = mood.lower()
    out = df[df["mood"].str.lower() == mood]

    if out.empty:
        return df.sample(5)

    return out.sample(min(5, len(out)))


# -------------------------------
# UI
# -------------------------------
def main():
    st.set_page_config(page_title="Mood2Music", page_icon="🎧", layout="wide")
    st.title("🎵 Mood2Music – Emotion-Based Song Recommender")

    st.write("This hybrid system uses **text emotion AI** to recommend songs.")

    df = load_music_data()
    text_model = load_text_classifier()

    st.subheader("📝 Enter text to detect mood")
    user_text = st.text_area("Type something describing your feelings...")

    if st.button("Analyze Mood"):
        with st.spinner("Detecting mood..."):
            mood = predict_text_emotion(user_text, text_model)

        st.success(f"Detected Mood: **{mood.upper()}**")

        st.subheader("🎶 Recommended Songs")
        recs = recommend_songs(df, mood)
        st.table(recs)

    st.write("---")
    st.write("Built with ❤️ using Streamlit, Transformers, Pandas & Matplotlib.")


if __name__ == "__main__":
    main()
