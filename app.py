# Import libraries
from gettext import install
from pdb import run
from pdb import run
from flask import app
import pip
import streamlit as st
import pandas as pd
import nltk
import string
import streamlit


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')

# Title
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and find out if it's Positive or Negative!")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\Vinayak\Downloads\archive\IMDB Dataset.csv")

df = load_data()

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Clean data
df['clean_review'] = df['review'].apply(preprocess_text)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

# Train model
model = LogisticRegression()
model.fit(X, y)

# User input
user_input = st.text_area("✍️ Enter your review here:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        clean_input = preprocess_text(user_input)
        vect_input = vectorizer.transform([clean_input])
        prediction = model.predict(vect_input)

        if prediction[0] == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😠 Negative Review")
    else:
        st.warning("Please enter some text!")
      

