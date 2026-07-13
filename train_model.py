import pandas as pd
import pickle
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("dataset.csv", sep="\t")

print("Dataset Shape:", df.shape)
print(df.head())


# ==========================
# Check Columns
# ==========================
print("Columns:", df.columns.tolist())


# Dataset should have:
# review -> text column
# sentiment -> target column

if "review" not in df.columns or "sentiment" not in df.columns:
    raise Exception(
        "Dataset must contain 'review' and 'sentiment' columns"
    )


# ==========================
# Split Data
# ==========================
X = df["review"]
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================
# Convert Text into Numbers
# ==========================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)


X_train_vector = vectorizer.fit_transform(X_train)

X_test_vector = vectorizer.transform(X_test)


# ==========================
# MLflow Experiment
# ==========================
mlflow.set_experiment(
    "Sentiment Analysis"
)


with mlflow.start_run():


    # ==========================
    # Train Model
    # ==========================
    model = LogisticRegression(
        max_iter=1000
    )


    model.fit(
        X_train_vector,
        y_train
    )


    # ==========================
    # Prediction
    # ==========================
    y_pred = model.predict(
        X_test_vector
    )


    # ==========================
    # Evaluation
    # ==========================
    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    print("\nModel Accuracy:")
    print(accuracy)


    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    # ==========================
    # MLflow Logging
    # ==========================
    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "max_features",
        5000
    )


    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    # ==========================
    # Save Model
    # ==========================
    with open(
        "sentiment_model.pkl",
        "wb"
    ) as f:
        pickle.dump(
            model,
            f
        )


    with open(
        "vectorizer.pkl",
        "wb"
    ) as f:
        pickle.dump(
            vectorizer,
            f
        )


    # ==========================
    # Save Model in MLflow
    # ==========================
    mlflow.sklearn.log_model(
        model,
        "sentiment_model"
    )


print("\n==============================")
print("Sentiment Model Trained Successfully!")
print("sentiment_model.pkl saved")
print("vectorizer.pkl saved")
print("MLflow logging completed")
print("==============================")