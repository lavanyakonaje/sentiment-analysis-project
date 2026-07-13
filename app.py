from fastapi import FastAPI
from pydantic import BaseModel
import pickle


# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API"
)


class Review(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is Running"
    }


@app.post("/predict")
def predict(review: Review):

    text_vector = vectorizer.transform(
        [review.text]
    )

    result = model.predict(
        text_vector
    )

    return {
        "review": review.text,
        "sentiment": result[0]
    }