import pandas as pd
import numpy as np
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Loading the Tensorflow Model for Prediction

model=load_model('model.h5')

with open('tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)


st.title('Twitter Tweets Sentiment Analysis')

tweet = st.text_area('Enter The Tweet')

if st.button('Predict Sentiment') and tweet.strip():
    sequences= tokenizer.texts_to_sequences([tweet])
    sequences=pad_sequences(sequences, padding = 'pre', maxlen=166)

    prediction = model.predict(sequences, verbose=0)
    predicted_class=np.argmax(prediction, axis=1)[0]

    sentiment_map={0 : 'Irrelevant',
                    1 : 'Negative',
                    2 : 'Neutral',
                    3 : 'Positive'
                    }
    st.write("Sentiment", sentiment_map[predicted_class])

