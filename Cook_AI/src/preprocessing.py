import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords once
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase, remove specials, tokenize
    text = re.sub('[^a-zA-Z ]', '', text.lower())
    words = text.split()
    # Remove stopwords and stem
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)
