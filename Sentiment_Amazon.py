import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Any

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax
import plotly.express as px

MAX_REVIEWS = 500
ROBERTA_MAX_LENGTH = 512
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "amazon-data" / "Reviews.csv"
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
REQUIRED_REVIEW_COLUMNS = ("Id", "Text", "Score", "Summary")
NLTK_RESOURCES = {
    "vader_lexicon": "sentiment/vader_lexicon.zip",
}
DEFAULT_ROBERTA_SCORES = {
    "roberta_neg": 0.33,
    "roberta_neu": 0.34,
    "roberta_pos": 0.33,
}


def ensure_nltk_resources() -> None:
    """Download NLTK resources only when they are missing locally."""
    for resource_name, resource_path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            downloaded = nltk.download(resource_name)
            if not downloaded:
                raise SystemExit(
                    "NLTK setup failed: could not download required resource "
                    f"'{resource_name}'. Check your network connection and NLTK data path."
                ) from None


def validate_review_schema(reviews: pd.DataFrame) -> None:
    """Ensure the loaded CSV has the columns used by this script."""
    missing_columns = [
        column for column in REQUIRED_REVIEW_COLUMNS if column not in reviews.columns
    ]
    if missing_columns:
        raise ValueError(
            "Reviews.csv is missing required columns: "
            f"{', '.join(missing_columns)}"
        )


def load_reviews(csv_path: Path, row_limit: int) -> pd.DataFrame:
    """Load a bounded number of reviews from the expected Kaggle CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(
            "Reviews.csv was not found. Download the Kaggle Amazon Fine Food "
            f"Reviews dataset and place it at: {csv_path}"
        )

    reviews = pd.read_csv(csv_path, nrows=row_limit)
    validate_review_schema(reviews)
    return reviews


def normalize_review_text(value: Any) -> str:
    """Convert missing review text to an empty string before sentiment scoring."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def polarity_scores_roberta(example: str) -> dict[str, float]:
    if not example:
        return DEFAULT_ROBERTA_SCORES.copy()

    encoded_text = tokenizer(
        example,
        return_tensors='tf',
        truncation=True,
        max_length=ROBERTA_MAX_LENGTH,
        padding='max_length',
    )
    output = model(encoded_text)[0]
    scores = output[0].numpy()
    scores = softmax(scores)
    return {'roberta_neg': scores[0], 'roberta_neu': scores[1], 'roberta_pos': scores[2]}


def load_roberta_components(model_name: str) -> tuple[Any, Any]:
    """Load the Hugging Face tokenizer and TensorFlow model with a clear setup error."""
    try:
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
        loaded_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as error:
        raise SystemExit(
            "RoBERTa setup failed: could not load the Hugging Face tokenizer/model. "
            "Check TensorFlow installation, transformers dependencies, network access, "
            "and the Hugging Face model cache/download state. "
            f"Original error: {error}"
        ) from None

    return loaded_tokenizer, loaded_model


try:
    df = load_reviews(DATA_FILE, MAX_REVIEWS)
    ensure_nltk_resources()
except (FileNotFoundError, ValueError) as error:
    raise SystemExit(f"Setup needed: {error}") from None

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()

# Use RoBERTa with TensorFlow
tokenizer, model = load_roberta_components(MODEL)

# Analyze dataset with VADER and RoBERTa
res = {}
failed_ids = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    myid = row.get('Id', f'row-{i}')
    try:
        text = normalize_review_text(row.get('Text'))
        vader_result = sia.polarity_scores(text)
        roberta_result = polarity_scores_roberta(text)
        combined = {f'vader_{k}': v for k, v in vader_result.items()}
        combined.update(roberta_result)
        res[myid] = combined
    except Exception as error:
        failed_ids.append((myid, str(error)))

if failed_ids:
    print("Some reviews could not be processed:")
    for failed_id, error in failed_ids:
        print(f"- Id {failed_id}: {error}")

if not res:
    raise SystemExit("No reviews were processed successfully. Check the errors above and try again.")

results_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left', on='Id')

# Rename columns for clarity
results_df.rename(columns={
    'roberta_neg': 'RoBERTa Negative Score',
    'roberta_neu': 'RoBERTa Neutral Score',
    'roberta_pos': 'RoBERTa Positive Score'
}, inplace=True)

# Create interactive 2D scatter plots with Plotly
fig1 = px.scatter(
    results_df,
    x='RoBERTa Negative Score',
    y='RoBERTa Positive Score',
    color='Score',
    hover_data=['Summary', 'Text'],
    title='Negative vs Positive Sentiment Analysis (RoBERTa) of Amazon Food Reviews'
)
fig1.show()

# fig2 = px.scatter(
#     results_df,
#     x='RoBERTa Negative Score',
#     y='RoBERTa Neutral Score',
#     color='Score',
#     hover_data=['Summary', 'Text'],
#     title='Negative vs Neutral Sentiment (RoBERTa)'
# )
# fig2.show()

# fig3 = px.scatter(
#     results_df,
#     x='RoBERTa Neutral Score',
#     y='RoBERTa Positive Score',
#     color='Score',
#     hover_data=['Summary', 'Text'],
#     title='Neutral vs Positive Sentiment (RoBERTa)'
# )
# fig3.show()

print("Successfully processed reviews and generated interactive 2D plots.")
