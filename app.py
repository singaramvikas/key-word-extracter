import streamlit as st
import spacy
from collections import Counter

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Function to preprocess and extract keywords using spaCy
def extract_keywords_spacy(text, top_n=15):
    doc = nlp(text)
    # Extract nouns, proper nouns, and verbs
    keywords = [token.text.lower() for token in doc if token.pos_ in ("NOUN", "PROPN", "VERB")]
    # Count the frequency of keywords
    keyword_freq = Counter(keywords)
    return keyword_freq.most_common(top_n)

# Streamlit App
st.title("Enhanced Job Description Keyword Extractor")

st.write("""
Paste the job description below, and this app will extract the most important keywords
to help you tailor your resume. This version uses advanced natural language processing (NLP) for better results.
""")

# Input text area for job description
job_description = st.text_area("Paste Job Description Here:", height=300)

# Extract keywords when the button is clicked
if st.button("Extract Keywords"):
    if job_description.strip():
        # Extract keywords using spaCy
        keywords = extract_keywords_spacy(job_description)
        st.subheader("Extracted Keywords (Word - Frequency):")
        for keyword, freq in keywords:
            st.write(f"{keyword} - {freq}")
    else:
        st.error("Please paste a job description to extract keywords.")
