from src.predict import predict_entities
from collections import defaultdict
import json

# ------------------------
# Terminal colors
# ------------------------
COLOR_MAP = {
    "PER": "\033[94m",   # Blue
    "ORG": "\033[92m",   # Green
    "LOC": "\033[93m",   # Yellow
    "MISC": "\033[95m"   # Purple
}
RESET = "\033[0m"


def main():
    print("\n🔤 Named Entity Recognition (NER)")
    print("Type a sentence and press Enter\n")

    # ------------------------
    # User input
    # ------------------------
    user_text = input("📝 Enter text: ").strip()

    if not user_text:
        print("❌ Empty input. Please enter valid text.")
        return

    # NER works better with capitalization
    text = user_text[0].upper() + user_text[1:]

    # ------------------------
    # Predict entities
    # ------------------------
    entities = predict_entities(text)

    print("\n🔍 Named Entities Found:\n")

    if not entities:
        print("No entities detected.")
        return

    # ------------------------
    # Display entities + stats
    # ------------------------
    label_stats = defaultdict(list)

    for ent in entities:
        color = COLOR_MAP.get(ent["label"], "")
        print(
            f"{color}Entity: {ent['entity']} | "
            f"Label: {ent['label']} | "
            f"Confidence: {ent['confidence']:.2f}{RESET}"
        )
        label_stats[ent["label"]].append(ent["confidence"])

    # ------------------------
    # Confidence summary
    # ------------------------
    print("\n📊 Confidence Summary:\n")
    for label, scores in label_stats.items():
        avg_score = sum(scores) / len(scores)
        print(f"{label}: Avg confidence = {avg_score:.2f}")

    # ------------------------
    # Save JSON
    # ------------------------
    save = input("\n💾 Save entities as JSON? (y/n): ").lower()
    if save == "y":
        with open("ner_output.json", "w") as f:
            json.dump(entities, f, indent=4)
        print("✅ Saved as ner_output.json")


if __name__ == "__main__":
    main()
