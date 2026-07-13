from fastapi import FastAPI
from pydantic import BaseModel
import pickle


# ==========================
# Load Model
# ==========================

model = pickle.load(
    open("sentiment_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)


# ==========================
# Create FastAPI App
# ==========================

app = FastAPI(
    title="Sentiment Analysis API"
)


# ==========================
# Input Format
# ==========================

class Review(BaseModel):
    text: str


# ==========================
# Home Route
# ==========================

@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API Running"
    }


# ==========================
# Prediction Route
# ==========================

@app.post("/predict")
def predict(review: Review):

    text_vector = vectorizer.transform(
        [review.text]
    )

    prediction = model.predict(
        text_vector
    )

    return {
        "review": review.text,
        "sentiment": prediction[0]
    }