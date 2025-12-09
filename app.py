from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="Sentiment API")

# -------------------------
# Load Model + Vectorizer
# -------------------------

# IMPORTANT:
# These files MUST be in the SAME folder as app.py:
#   - ReviewModel4.pkl
#   - tfidf_vectorizer (1).pkl

with open(r"ReviewModel4.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"tfidf_vectorizer (1).pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------
# Input Schema
# -------------------------
class ReviewInput(BaseModel):
    text: str

# -------------------------
# Prediction Mappings
# -------------------------
meaning_map = {
    0: "Customer is VERY angry.",
    1: "Customer is neutral. Follow up required.",
    2: "Customer is VERY happy."
}

action_map = {
    0: "Call customer immediately to resolve the issue.",
    1: "Follow up soon to keep the customer.",
    2: "No action needed — customer is happy."
}

# -------------------------
# Routes
# -------------------------

@app.get("/")
def home():
    return {"message": "Sentiment API is running!"}

@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    text = data.text

    # Vectorize text
    vec = vectorizer.transform([text])

    # Predict class
    pred = model.predict(vec)[0]

    return {
        "prediction": int(pred),
        "meaning": meaning_map[pred],
        "action": action_map[pred]
    }
