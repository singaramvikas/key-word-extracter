import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return words

def extract_keywords_frequency(words, top_n=10):
    word_counts = Counter(words)
    return [word for word, _ in word_counts.most_common(top_n)]

def extract_keywords_tfidf(text, top_n=10):
    vectorizer = TfidfVectorizer(max_features=top_n)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

st.title("Job Description Keyword Extractor")
st.write("Paste the job description below, and this app will extract the most important keywords to help you tailor your resume.")

job_description = st.text_area("Paste Job Description Here:", height=300)
method = st.radio("Select Keyword Extraction Method:", ("Frequency Analysis", "TF-IDF"))

if st.button("Extract Keywords"):
    if job_description.strip():
        if method == "Frequency Analysis":
            cleaned_text = preprocess_text(job_description)
            keywords = extract_keywords_frequency(cleaned_text)
        else:
            keywords = extract_keywords_tfidf(job_description)
        st.subheader("Extracted Keywords:")
        st.write(", ".join(keywords))
    else:
        st.error("Please paste a job description to extract keywords.")
