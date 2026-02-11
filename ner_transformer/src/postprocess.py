def clean_entities(entities):
    """
    Merge duplicate entities and keep max confidence
    """
    entity_map = {}

    for ent in entities:
        key = (ent["word"], ent["entity_group"])

        if key not in entity_map:
            entity_map[key] = ent["score"]
        else:
            entity_map[key] = max(entity_map[key], ent["score"])

    cleaned = []
    for (word, label), score in entity_map.items():
        cleaned.append({
            "entity": word,
            "label": label,
            "confidence": round(score, 2)
        })

    return cleaned
