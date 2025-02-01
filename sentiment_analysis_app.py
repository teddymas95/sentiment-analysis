#!pip install streamlit transformers pandas
import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Sample dataset (replace with your own data)
data = {
    "text": [
        "This is a very good product.",
        "I am not happy with this service.",
        "Neutral opinion.",
        "Excellent! I love it.",
        "This is terrible."
    ]
}
df = pd.DataFrame(data)

# Streamlit app
st.title("Sentiment Analysis Demo")

# User input
user_input = st.text_input("Enter text:")

# Analyze user input
if user_input:
    result = sentiment_analyzer(user_input)[0]
    st.write(f"**Sentiment:** {result['label']}")
    st.write(f"**Score:** {result['score']:.2f}")

# Display sample data
st.header("Sample Data")
st.dataframe(df)

# Analyze sample data
if st.button("Analyze Sample Data"):
    df['sentiment'] = df['text'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
    df['score'] = df['text'].apply(lambda x: sentiment_analyzer(x)[0]['score'])
    st.dataframe(df)
