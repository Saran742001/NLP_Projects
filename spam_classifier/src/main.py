df = load_dataset("data/spam.csv")

df["clean_message"] = df["message"].apply(clean_text)

# Encode labels
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X = df["clean_message"]
y = df["label_num"]

