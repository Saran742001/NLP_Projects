# predict.py
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------------------
# Load Model and Vectorizer
# --------------------------
try:
    model = joblib.load("spam_nb_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("✓ Model and vectorizer loaded successfully!\n")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please run main.py first to generate model and vectorizer.")
    exit()

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
# Add New Messages Here
# --------------------------
new_messages = [
    "Congratulations! You won a free iPhone!",
    "Hey, are we still meeting for lunch?",
    "Claim your prize by clicking this link now!",
    "Don't forget to submit your assignment today."
]

# --------------------------
# Predict
# --------------------------
print("Predictions for new messages:\n")
for msg in new_messages:
    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    print(f"Message: {msg}\nPredicted label: {'Spam' if pred[0]==1 else 'Ham'}\n")
