import streamlit as st
import numpy as np
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model("emotion_model.h5")

# Function for preprocessing input text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos='v') for word in filtered_tokens]
    # Join tokens into a single string
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Function to make predictions
def make_prediction(text):
    # Preprocess input text
    preprocessed_text = preprocess_text(text)
    # Tokenize and pad the input
    tokenized_text = tokenizer.texts_to_sequences([preprocessed_text])
    padded_text = pad_sequences(tokenized_text, maxlen=maxlen, padding='pre')
    # Make prediction
    prediction = model.predict(padded_text)
    # Get the predicted emotion label
    predicted_label = np.argmax(prediction)
    return predicted_label

# Streamlit UI
def main():
    st.title("Emotion Classifier")
    # Text input for user
    user_input = st.text_input("Enter your text here:")
    if st.button("Predict"):
        if user_input:
            # Make prediction
            prediction = make_prediction(user_input)
            # Map prediction label to emotion
            emotions = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
            predicted_emotion = emotions[prediction]
            st.write(f"Predicted Emotion: {predicted_emotion}")

if __name__ == "__main__":
    main()
