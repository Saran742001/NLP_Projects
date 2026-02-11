from src.load_data import load_dataset
from src.preprocessing import clean_text

from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import os



def visualize_topics(lda_model, dictionary):
    """
    Create WordCloud for each topic
    """
    for topic_id in range(lda_model.num_topics):
        words = dict(lda_model.show_topic(topic_id, topn=30))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate_from_frequencies(words)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic {topic_id}", fontsize=14)
        plt.show()


def predict_topic(text, lda_model, dictionary):
    clean = clean_text(text)
    tokens = clean.split()
    bow = dictionary.doc2bow(tokens)

    if len(bow) == 0:
        return None, None

    topics = lda_model.get_document_topics(bow)

    if not topics:
        return None, None

    topic_id, confidence = max(topics, key=lambda x: x[1])
    return topic_id, confidence



def main():
    # ------------------------
    # Load dataset
    # ------------------------
    df = load_dataset()

    # Clean text
    df["clean_content"] = df["content"].apply(clean_text)

    # Tokenize
    texts = df["clean_content"].apply(lambda x: x.split())

    # Dictionary & Corpus
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # ------------------------
    # Train LDA model
    # ------------------------
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=5,
        random_state=42,
        passes=10,
        alpha="auto"
    )

    # Print topics
    print("\n🧠 Topics:\n")
    for idx, topic in lda_model.print_topics(num_words=8):
        print(f"Topic {idx}: {topic}")

    # ------------------------
    # Assign dominant topic
    # ------------------------
    def get_dominant_topic(bow):
        topic_probs = lda_model.get_document_topics(bow)
        return max(topic_probs, key=lambda x: x[1])[0]

    df["topic"] = [get_dominant_topic(bow) for bow in corpus]

    print("\n📄 Sample Document → Topic Mapping:\n")
    print(df[["content", "topic"]].head())

    # ------------------------
    # STEP-08: Visualize Topics
    # ------------------------
    print("\n🎨 Generating Topic WordClouds...\n")
    visualize_topics(lda_model, dictionary)



    # ------------------------
    # STEP-09: Coherence Score ✅ FIXED
    # ------------------------
    print("\n🔍 Calculating coherence score...")

    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts.tolist(),   # ✅ CRITICAL FIX
        dictionary=dictionary,
        coherence="c_v"
    )

    coherence_score = coherence_model.get_coherence()

    print("📊 Coherence Score:", round(coherence_score, 3))


    os.makedirs("models", exist_ok=True)

    lda_model.save("models/lda_model.model")
    dictionary.save("models/dictionary.dict")

    print("\n💾 LDA model saved to models/lda_model.model")
    print("💾 Dictionary saved to models/dictionary.dict")


    # ------------------------
    # USER INTERACTION LOOP
    # ------------------------
    print("\n🧠 Topic Modeling Interactive Mode")
    print("Type 'exit' to quit\n")

    while True:
        user_text = input("Enter text: ")

        if user_text.lower() == "exit":
            print("👋 Exiting. Bye!")
            break

        topic_id, confidence = predict_topic(
            user_text,
            lda_model,
            dictionary
        )

        if topic_id is None:
            print("⚠️ Could not predict topic\n")
            continue

        print(f"\n🔍 Predicted Topic: {topic_id}")
        print(f"📈 Confidence: {confidence:.2f}")

        choice = input("\nDo you want coherence score? (yes/no): ").lower()
        if choice == "yes":
            print("📊 Model Coherence Score:", round(coherence_score, 3))

        print("-" * 50)



    # ------------------------
    # Predict topic for new text
    # ------------------------
    sample_text = "The government announced new economic reforms today"


    topic_id, confidence = predict_topic(
    sample_text,
    lda_model,
    dictionary
    )

    print("\n🆕 New Text Topic Prediction:")
    print("Text:", sample_text)
    if topic_id is None:
     print("⚠️ No topic could be assigned to this text")
    else:
     print(f"Predicted Topic: {topic_id} (confidence={confidence:.2f})")


if __name__ == "__main__":
    main()
