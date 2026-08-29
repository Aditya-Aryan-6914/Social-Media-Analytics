from .config import LABEL_MAP
from .load_models import sentiment_pipeline
from .preprocessing import clean_text, clean_texts

# Format the predictions returned by the sentiment pipeline into a structured dictionary.
def _format_prediction(predictions: list[dict]) -> dict:
    best_prediction = max(
        predictions,
        key=lambda prediction: prediction["score"]
    )

    scores = {
        p["label"]: float(p["score"])
        for p in predictions
    }

    return {
        "sentiment": best_prediction["label"],
        "confidence": float(best_prediction["score"]),
        "scores": scores,
    }


# def predict_sentiment(text: str) -> dict:

#     # Predict sentiment for a single social-media post.
#     cleaned_text = clean_text(text)

#     predictions = sentiment_pipeline(
#         cleaned_text,
#         truncation=True,
#         max_length=512,
#     )

#     return _format_prediction(predictions)


def predict_sentiment_batch(texts: list[str]) -> list[dict]:

    # Predict sentiment for multiple social-media posts.
    if not texts:
        return []

    cleaned_texts = clean_texts(texts)

    predictions_batch = sentiment_pipeline(
        cleaned_texts,
        truncation=True,
        max_length=512,
        batch_size=16,
    )

    #returns list of dicts with sentiment, confidence and scores for each text
    return [
        _format_prediction(predictions)
        for predictions in predictions_batch
    ]