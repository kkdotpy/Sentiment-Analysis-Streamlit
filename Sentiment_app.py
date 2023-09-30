import streamlit as st
import pandas as pd
import numpy as np
from SentimentAnalysisScript import sentiment_finder

# Set page title and background style
st.title("Real-Time Sentiment Analysis")

# Define a custom HTML template for styling
custom_html = """
<div style="background-color: #007BFF; padding: 20px; border-radius: 10px;">
    <h1 style="color: white; text-align: center;">Real-Time Sentiment Analysis</h1>
</div>
"""
# Apply the custom HTML template
st.markdown(custom_html, unsafe_allow_html=True)

# Create a text input field for user input
text = st.text_input("Enter your text for sentiment analysis:", "")

# Create a button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if text:
        result = sentiment_finder(text)  # Use the sentiment_finder function directly
        st.success(f"Sentiment Analysis Result: {result}")
    else:
        st.warning("Please enter text for analysis.")
