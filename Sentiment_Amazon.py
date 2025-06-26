import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax
import plotly.express as px

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load dataset
df = pd.read_csv('amazon-data/Reviews.csv')
df = df.head(500)

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
res = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

# Use RoBERTa with TensorFlow
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(example):
    if example is None or pd.isna(example):
        return {'roberta_neg': 0.33, 'roberta_neu': 0.34, 'roberta_pos': 0.33}
    text = str(example)
    encoded_text = tokenizer(text, return_tensors='tf', truncation=True, max_length=512, padding='max_length')
    output = model(encoded_text)[0]
    scores = output[0].numpy()
    scores = softmax(scores)
    return {'roberta_neg': scores[0], 'roberta_neu': scores[1], 'roberta_pos': scores[2]}

# Analyze dataset with VADER and RoBERTa
res = {}
failed_ids = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        roberta_result = polarity_scores_roberta(text)
        combined = {f'vader_{k}': v for k, v in vader_result.items()}
        combined.update(roberta_result)
        res[myid] = combined
    except Exception as e:
        failed_ids.append(myid)

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