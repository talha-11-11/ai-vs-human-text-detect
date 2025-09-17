import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
from sentence_transformers import SentenceTransformer

# ---------- Load Dataset ----------
df = pd.read_csv("data/text/text_data.csv")
df["text"] = df["text"].astype(str)

# ---------- Stylometric Features ----------
def stylometric_features(text):
    words = text.split()
    num_words = len(words)
    num_chars = len(text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_len = np.mean([len(s.split()) for s in re.split(r'[.!?]', text) if s]) if text else 0
    return [num_words, num_chars, avg_word_len, avg_sentence_len]

style_features = np.array([stylometric_features(t) for t in df["text"]])

# ---------- Embeddings ----------
print("🔄 Generating embeddings (this may take some time)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(df["text"].tolist(), convert_to_numpy=True, show_progress_bar=True)

# ---------- Combine Features ----------
X_full = np.hstack([embeddings, style_features])
y = df["label"].values

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Model (Logistic Regression for interpretability) ----------
pipeline = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
])

print("🚀 Training hybrid model...")
pipeline.fit(X_train, y_train)

# ---------- Evaluate ----------
y_pred = pipeline.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Save ----------
joblib.dump(pipeline, "models/text_classifier.pkl")
joblib.dump(embedder, "models/embedder.pkl")

print("\n🎉 Hybrid model and embedder saved in models/")
