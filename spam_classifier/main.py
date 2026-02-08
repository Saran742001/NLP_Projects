# main.py
import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------
# Preprocessing
# --------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1", on_bad_lines='skip')
df = df[['label','message']]  # Keep only relevant columns
df.dropna(inplace=True)
df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham':0, 'spam':1})

X = df['clean_message']
y = df['label_num']

# --------------------------
# Split Dataset
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# TF-IDF Vectorization
# --------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------
# Train Model
# --------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --------------------------
# Evaluate Model
# --------------------------
y_pred = model.predict(X_test_vec)
accuracy = model.score(X_test_vec, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ham','spam']))

# --------------------------
# Save Model & Vectorizer
# --------------------------
joblib.dump(model, "spam_nb_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\n✅ Model and vectorizer saved successfully!")
