import re
import pickle
import os

# -----------------------------
# Text Cleaning Function
# -----------------------------
def get_clean_text(text):
    text = text.lower()                                     # normalize
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)              # remove @mentions
    text = re.sub(r'#', '', text)                           # remove hashtags (# only)
    text = re.sub(r'RT[\s]+', '', text)                     # remove RT
    text = re.sub(r'https?:\/\/\S+', '', text)              # remove URLs
    text = re.sub(r'(\\u[a-z0-9]+)', '', text)              # remove unicode escape \uxxxx
    text = re.sub(r'"', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'https?:?', '', text)
    text = re.sub(r'href', '', text)
    text = re.sub(r'\s+', ' ', text).strip()                # remove extra spaces
    return text


# -----------------------------
# Save Function as Pickle
# -----------------------------
artifact_path = "artifacts/clean_text.pkl"
os.makedirs("artifacts", exist_ok=True)

with open(artifact_path, 'wb') as f:
    pickle.dump(get_clean_text, f)

print(f"[INFO] Clean function saved → {artifact_path}")


# -----------------------------
# Load Function from Pickle
# (As you requested)
# -----------------------------
text_clean_function = pickle.load(open(r'artifacts/clean_text.pkl', 'rb'))
