from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load model once (important)
MODEL_NAME = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  # 🔥 groups entities correctly
)

def predict_entities(text, confidence_threshold=0.6):
    """
    Predict named entities with confidence filtering
    """
    results = ner_pipeline(text)

    entities = []

    for ent in results:
        if ent["score"] >= confidence_threshold:
            entities.append({
                "entity": ent["word"],
                "label": ent["entity_group"],
                "confidence": round(ent["score"], 2)
            })

    return entities
