# main.py (at project root, outside src/)
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
# Preprocessing function
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
# Load dataset
# --------------------------
# The CSV has mismatched header (comma) and data (tab), so skip the header row
df = pd.read_csv("data/spam.csv", encoding="latin-1", sep="\t", skiprows=1, header=None, on_bad_lines="skip")
df.columns = ['label', 'message']

print(f"Total rows loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"First 3 rows:\n{df.head(3)}")
print(f"Data types: {df.dtypes.to_dict()}")

# Remove rows with missing labels or messages
print(f"Rows before dropna: {len(df)}")
df = df.dropna(subset=["label", "message"])
print(f"Rows after dropna: {len(df)}")

# Strip whitespace from labels and convert to lowercase
df["label"] = df["label"].str.strip().str.lower()

print(f"After stripping - Unique labels: {df['label'].unique()}")
print(f"Label counts:\n{df['label'].value_counts()}")

df["clean_message"] = df["message"].apply(clean_text)

# Remove rows with empty cleaned messages
df = df[df["clean_message"].str.len() > 0]
print(f"Rows after removing empty messages: {len(df)}")

# Encode labels
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Remove rows where label_num is NaN (labels that weren't 'ham' or 'spam')
print(f"Rows before removing NaN labels: {len(df)}")
df = df.dropna(subset=["label_num"])
print(f"Rows after removing NaN labels: {len(df)}")
df["label_num"] = df["label_num"].astype(int)

print(f"Rows after cleaning: {len(df)}")
print(f"Label distribution:\n{df['label_num'].value_counts()}")

X = df["clean_message"]
y = df["label_num"]

# --------------------------
# Split dataset
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Vectorize
# --------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------
# Train Naive Bayes
# --------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# --------------------------
# Evaluation
# --------------------------
accuracy = model.score(X_test_vec, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred, target_names=["ham", "spam"])
print("\nClassification Report:")
print(report)

# --------------------------
# Save model and vectorizer
# --------------------------
joblib.dump(model, "spam_nb_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\n✅ Model and vectorizer saved successfully!")

# --------------------------
# Predict new messages
# --------------------------
new_messages = [
    "Congratulations! You won a free iPhone!",      # should be Spam
    "Hey, are we still meeting for lunch or not?",         # should be Ham
    "Claim your prize by clicking this link now in the description!",  # should be Spam
    "Don't forget to submit your assignment within a week." # should be Ham
]

for msg in new_messages:
    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    print(f"\nMessage: {msg}\nPrediction: {'Spam' if pred[0]==1 else 'Ham'}")
