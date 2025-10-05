import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample FAQ data
faq_data = {
    "questions": [
        "What is CodeAlpha?",
        "How do I submit my internship tasks?",
        "When will I receive my certificate?",
        "Can I choose multiple domains?",
        "Is a demo video mandatory?"
    ],
    "answers": [
        "CodeAlpha is a virtual internship platform offering projects in various domains.",
        "You must submit your tasks via the official form between Sept 20 – Oct 20.",
        "Certificates and LORs are issued on Oct 21 if tasks are submitted correctly.",
        "Yes, you can choose multiple domains, but you only need to complete 2–3 tasks from one.",
        "No, demo videos are optional. GitHub + LinkedIn post are enough."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(faq_data)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['questions'])

# Streamlit UI
st.title("🤖 FAQ Chatbot")
user_question = st.text_input("Ask a question:")

if user_question:
    user_vec = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_match = similarity.argmax()
    st.success(f"Answer: {df['answers'][best_match]}")# ChatBox-for-Faqs--Streamlit-App
