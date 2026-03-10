# Stub for future implementation
def predict_prep_time(recipe_name, ingredients, instructions):
    # Regression model or simple logic for prep time estimation
    words_count = len(instructions.split())
    estimated_time = max(10, words_count // 5)
    return f"{estimated_time} mins"

if __name__ == "__main__":
    print(predict_prep_time("Tomato Soup", "Tomato, Onion", "Boil and blend."))
