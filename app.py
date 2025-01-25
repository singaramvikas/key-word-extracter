import streamlit as st
import spacy
from collections import Counter

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        # Load the spaCy model
        return spacy.load("en_core_web_sm")
    except OSError:
        # If the model is not found, download and load it
        st.warning("Downloading 'en_core_web_sm' model...")
        spacy.cli.download("en_core_web_sm")
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

# Function to extract phrases (noun chunks) using spaCy
def extract_phrases_spacy(text, top_n=10):
    doc = nlp(text)
    # Extract noun chunks (phrases)
    phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
    # Count the frequency of phrases
    phrase_freq = Counter(phrases)
    return phrase_freq.most_common(top_n)

# Streamlit App
st.title("Enhanced Job Description Keyword Extractor")

st.write("""
Paste the job description below, and this app will extract the most important keywords
and phrases to help you tailor your resume. This version uses advanced natural language processing (NLP) for better results.
""")

# Input text area for job description
job_description = st.text_area("Paste Job Description Here:", height=300)

# Select keyword or phrase extraction
method = st.radio("Select Extraction Method:", ("Keywords", "Phrases"))

# Extract keywords or phrases when the button is clicked
if st.button("Extract"):
    if job_description.strip():
        if method == "Keywords":
            # Extract keywords
            keywords = extract_keywords_spacy(job_description)
            st.subheader("Extracted Keywords (Word - Frequency):")
            for keyword, freq in keywords:
                st.write(f"{keyword} - {freq}")
        else:
            # Extract phrases
            phrases = extract_phrases_spacy(job_description)
            st.subheader("Extracted Phrases (Phrase - Frequency):")
            for phrase, freq in phrases:
                st.write(f"{phrase} - {freq}")
    else:
        st.error("Please paste a job description to extract keywords or phrases.")
