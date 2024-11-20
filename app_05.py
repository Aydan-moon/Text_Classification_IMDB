import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained SVM model
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load the TF-IDF vectorizer used in training
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load spaCy model for text cleaning
nlp = spacy.load("en_core_web_sm")

# Text preprocessing function
def clean_review(review):
    review = re.sub(r'<.*?>', '', review)  # Remove HTML tags
    review = re.sub(r'[^a-zA-Z\s]', '', review)  # Remove non-alphabetic characters
    review = review.lower()  # Convert to lowercase
    review = re.sub(r'\s+', ' ', review)  # Remove extra whitespaces

    doc = nlp(review)
    cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(cleaned_tokens)

# Streamlit app code
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="wide")
st.sidebar.title("IMDB Sentiment Classifier")
st.sidebar.write(
    """
    This application classifies IMDB movie reviews as **positive** or **negative** using an SVM model trained on the IMDB dataset. 
    Enter a review below, or upload a file with multiple reviews for batch prediction.
    """
)
st.sidebar.write("### Model Information")
st.sidebar.write("**Model:** Support Vector Machine (SVM)")
st.sidebar.write("**Vectorizer:** TF-IDF")
st.sidebar.write("**Performance Metrics:**")
st.sidebar.write("- Accuracy: 88.45%")
st.sidebar.write("- F1-Score: 88.74%")

# Main title and instructions
st.title("IMDB Movie Review Sentiment Classifier")
st.write("This app classifies IMDB movie reviews as positive or negative based on the review text.")

# Input for single review prediction
st.write("### Single Review Prediction")
review_text = st.text_area("Enter a movie review:", "")

if st.button("Predict Sentiment"):
    if review_text:
        # Preprocess and vectorize the review
        cleaned_review = clean_review(review_text)
        review_vector = tfidf_vectorizer.transform([cleaned_review])
        
        # Predict sentiment
        prediction = svm_model.predict(review_vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        # Display the result with custom feedback
        if sentiment == "Positive":
            st.success("🎉 The review has a **Positive** sentiment!")
        else:
            st.warning("😞 The review has a **Negative** sentiment.")
    else:
        st.write("Please enter a review to analyze.")

# Batch Prediction Section
st.write("---")
st.write("### Batch Prediction on Uploaded Reviews")
uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"])

if uploaded_file:
    try:
        # Load and display the file
        df = pd.read_csv(uploaded_file)
        if 'review' in df.columns:
            # Preprocess reviews
            df['cleaned_review'] = df['review'].apply(clean_review)
            review_vectors = tfidf_vectorizer.transform(df['cleaned_review'])
            
            # Predict sentiment for each review
            df['predicted_sentiment'] = svm_model.predict(review_vectors)
            df['predicted_sentiment'] = df['predicted_sentiment'].map({1: 'Positive', 0: 'Negative'})
            
            st.write("### Prediction Results")
            st.write(df[['review', 'predicted_sentiment']])

            # Display sentiment distribution in a bar chart
            st.write("### Sentiment Distribution")
            sentiment_counts = df['predicted_sentiment'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax)
            ax.set_title("Sentiment Distribution")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.error("The uploaded file does not have a 'review' column.")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
