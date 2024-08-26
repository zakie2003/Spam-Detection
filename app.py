import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
import re
import joblib

# # Initialize the RandomUnderSampler
# enn = RandomUnderSampler()

# # Load and preprocess the data
# data = pd.read_csv("final.csv")
# result = data["v1"]
# data = data.drop(["v1"], axis=1)

# # Initialize the TF-IDF Vectorizer and fit it on the data
# tfidf_vectorizer = TfidfVectorizer()
# array = tfidf_vectorizer.fit_transform(data["v2"])
# data = array.toarray()

# # Perform undersampling
# data, result = enn.fit_resample(data, result)

# Define preprocessing functions
def html_tag_removal(text):
    html_tags_pattern = r'<.*?>'
    text_without_html_tags = re.sub(html_tags_pattern, '', text)
    return text_without_html_tags

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def input_preprocess(txt):
    txt = txt.lower()
    txt = remove_urls(txt)
    txt = html_tag_removal(txt)
    txt = re.sub(r'[^\w\s]', '', txt)
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    array = tfidf_vectorizer.transform([txt])
    return array.toarray()

# Streamlit UI
st.title("Email Spam Detection")

# User input text box
input_text = st.text_area("Enter your text here:", "Hi Zakie, How Are You?")

# Preprocess the input text and display the result
if st.button("Preprocess Text"):
    preprocessed_text = input_preprocess(input_text)
    from_joblib = joblib.load('Spam_Model.pkl')
    pred=from_joblib.predict(preprocessed_text)
    if(pred[0]==1):
        st.write(f"The Given Message is Spam!")
    else:
        st.write(f"The Given Message is not Spam!")
