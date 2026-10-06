# Twitter Tweets Sentiment Analysis

A Twitter sentiment analysis project built using Python, NLP, TensorFlow, and SimpleRNN. The model classifies tweets into four categories: Positive, Negative, Neutral, and Irrelevant.

## Technologies Used

* Python
* Pandas
* NumPy
* TensorFlow / Keras
* Scikit-learn
* NLP
* SimpleRNN
* Streamlit

## Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Label Encoding
   ↓
Tokenization
   ↓
Sequence Padding
   ↓
Train / Validation Split
   ↓
Embedding
   ↓
SimpleRNN
   ↓
Dropout
   ↓
Dense + Softmax
   ↓
Sentiment Prediction
```

## Model

The project uses a SimpleRNN model with an Embedding layer, SimpleRNN, Dropout, and a Dense output layer with Softmax activation.

The four output classes are:

```text
0 - Irrelevant
1 - Negative
2 - Neutral
3 - Positive
```

## Streamlit Application

The trained model and tokenizer are saved as:

```text
model.h5
tokenizer.pkl
```

Run the application using:

```bash
streamlit run main.py
```

Enter a tweet in the application to get its predicted sentiment.

## Project Structure

```text
Twitter_Project/
├── main.py
├── model.h5
├── tokenizer.pkl
├── twitter_training.csv
├── Twitter_project.ipynb
└── README.md
```

## Learning Outcomes

This project covers NLP preprocessing, tokenization, sequence padding, word embeddings, SimpleRNN, dropout, model training, evaluation, model saving, and Streamlit deployment.

## Author

Ganesh Kaithoju
