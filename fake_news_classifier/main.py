from src.load_data import load_dataset
from src.preprocessing import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import joblib

def train_and_save():
    df = load_dataset("data/fake_news.csv")

    df["content"] = df["title"] + " " + df["text"]
    df["clean_content"] = df["content"].apply(clean_text)
    df["label_num"] = df["label"].map({"REAL": 0, "FAKE": 1})

    X = df["clean_content"]
    y = df["label_num"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n🎯 Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "fake_news_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    print("\n✅ Model & Vectorizer saved")

def predict_news():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    print("\n📰 Fake News Detection (type 'exit' to quit)\n")

    while True:
        user_input = input("Enter news text: ")

        if user_input.lower() == "exit":
            print("👋 Exiting prediction mode")
            break

        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            print("🚨 Prediction: FAKE NEWS\n")
        else:
            print("✅ Prediction: REAL NEWS\n")

if __name__ == "__main__":
    train_and_save()
    predict_news()
