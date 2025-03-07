import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Loading the Imdb dataset's word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# loading the pre-trained model
model = load_model('imdb_rnn.keras')

## function to decode the reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## function to preprocess the user text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review

## Prediction function
def predict_sentiment(review):
    pp_inp = preprocess_text(review)
    prediction = model.predict(pp_inp)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

## Streamlit app design
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    
    sentiment, score = predict_sentiment(user_input)
    #st.write(f'Review: {user_input}')
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else: 
    st.write('Please enter a movie review.')