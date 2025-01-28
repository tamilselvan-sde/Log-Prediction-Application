import streamlit as st
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# Load the saved RNN model
@st.cache_resource
def load_rnn_model():
    return load_model('rnn_log_model.h5')

# Load log embeddings and flattened logs
@st.cache_resource
def load_log_data():
    with open('log_data.pkl', 'rb') as f:
        return pickle.load(f)

# Reload the SentenceTransformer model
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to calculate log embedding
def calculate_log_embedding(log, sentence_model):
    words = log.split()
    word_embeddings = sentence_model.encode(words)
    return np.mean(word_embeddings, axis=0)

# Function to predict the next log
def predict_next_log(input_log, model_rnn, log_embeddings, flattened_logs, sentence_model):
    # Preprocess and calculate the embedding for the input log
    input_embedding = calculate_log_embedding(input_log, sentence_model).reshape(1, 1, -1)

    # Predict the next log embedding
    predicted_embedding = model_rnn.predict(input_embedding)

    # Find the closest log in the training set for the predicted embedding
    similarities = cosine_similarity(predicted_embedding, log_embeddings)
    predicted_index = np.argmax(similarities)

    # Return the predicted log
    return flattened_logs[predicted_index]

# Load resources
model_rnn = load_rnn_model()
data = load_log_data()
log_embeddings = data['log_embeddings']
flattened_logs = data['flattened_logs']
sentence_model = load_sentence_model()

# Streamlit UI
st.title("Log Prediction Application")

# Display a list of logs in the UI
logs = [
    "Application started",
    "User logged in",
    "API request made to /endpoint",
    "Database query executed",
    "Response returned to user",
    "User performed action",
    "Another API request to /other-endpoint",
    "Cache updated",
    "User logged out",
    "Application ended successfully"
]

st.subheader("Sample Logs:")
for log in logs:
    st.markdown(f"- {log}")

input_log = st.text_input("Enter a log entry:", "User performed action")

if st.button("Predict Next Log"):
    with st.spinner("Predicting next log..."):
        predicted_log = predict_next_log(input_log, model_rnn, log_embeddings, flattened_logs, sentence_model)
    st.success(f"Predicted next log: {predicted_log}")
