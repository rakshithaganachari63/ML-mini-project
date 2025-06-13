# ================ SPAM EMAIL CLASSIFIER ================
# Complete implementation with training, evaluation, and prediction

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download NLTK data (first-time only)
nltk.download('stopwords')

# ================ 1. Data Loading & Preprocessing ================
def load_data(filepath='spam.csv'):
    """Load and preprocess the dataset"""
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]  # Keep only label and text columns
    df.columns = ['label', 'text']
    
    # Convert labels to binary (0=ham, 1=spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def clean_text(text):
    """Text preprocessing: lowercase, remove special chars, stemming"""
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special chars
    text = text.lower().split()
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# ================ 2. Feature Engineering ================
def create_features(df):
    """Create TF-IDF features"""
    # Text cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Bag-of-Words
    count_vect = CountVectorizer(max_features=5000)
    X_train_counts = count_vect.fit_transform(X_train)
    
    # TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    return X_train, X_test, y_train, y_test, count_vect, tfidf_transformer

# ================ 3. Model Training ================
def train_model(X_train_tfidf, y_train):
    """Train and return a Naive Bayes classifier"""
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model

# ================ 4. Evaluation ================
def evaluate_model(model, count_vect, tfidf_transformer, X_test, y_test):
    """Evaluate model performance"""
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_pred = model.predict(X_test_tfidf)
    
    print("\nEvaluation Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================ 5. Prediction Function ================
def predict_spam(model, count_vect, tfidf_transformer, email_text):
    """Predict if a message is spam (1) or ham (0)"""
    cleaned_text = clean_text(email_text)
    email_counts = count_vect.transform([cleaned_text])
    email_tfidf = tfidf_transformer.transform(email_counts)
    return model.predict(email_tfidf)[0]

# ================ 6. Save/Load Model ================
def save_artifacts(model, count_vect, tfidf_transformer):
    """Save model and vectorizers"""
    joblib.dump(model, 'spam_classifier.joblib')
    joblib.dump(count_vect, 'count_vectorizer.joblib')
    joblib.dump(tfidf_transformer, 'tfidf_transformer.joblib')
    print("Model artifacts saved successfully")

def load_artifacts():
    """Load saved artifacts"""
    model = joblib.load('spam_classifier.joblib')
    count_vect = joblib.load('count_vectorizer.joblib')
    tfidf_transformer = joblib.load('tfidf_transformer.joblib')
    return model, count_vect, tfidf_transformer

# ================ MAIN EXECUTION ================
if __name__ == "__main__":
    # Step 1: Load data
    print("Loading data...")
    df = load_data()
    
    # Step 2: Create features
    print("Creating features...")
    X_train, X_test, y_train, y_test, count_vect, tfidf_transformer = create_features(df)
    
    # Step 3: Train model
    print("Training model...")
    model = train_model(tfidf_transformer.transform(count_vect.transform(X_train)), y_train)
    
    # Step 4: Evaluate
    evaluate_model(model, count_vect, tfidf_transformer, X_test, y_test)
    
    # Step 5: Save artifacts
    save_artifacts(model, count_vect, tfidf_transformer)
    
    # Step 6: Demo prediction
    test_messages = [
        "WIN A FREE IPHONE TODAY! Click here to claim your prize!",
        "Hey John, just checking in about our meeting tomorrow at 2pm",
        "Your account has been compromised. Verify your password now!"
    ]
    
    print("\nDemo Predictions:")
    for msg in test_messages:
        prediction = predict_spam(model, count_vect, tfidf_transformer, msg)
        print(f"\nMessage: {msg[:50]}...\nPrediction: {'SPAM' if prediction == 1 else 'HAM'}")
