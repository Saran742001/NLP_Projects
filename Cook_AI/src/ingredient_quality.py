# Stub for future implementation
def predict_quality(ingredients):
    return "High Quality" if len(ingredients.split(",")) > 3 else "Moderate Quality"

if __name__ == "__main__":
    print(predict_quality("Tomato, Garlic, Onion, Basil"))
