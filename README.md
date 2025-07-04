🎬 RNN Movie Review Sentiment Analysis

A simple Recurrent Neural Network (RNN) model built with TensorFlow/Keras to classify IMDb movie reviews as positive or negative.

👉 Live Demo: https://rnn-movie-review-sentiment-analysis.streamlit.app/

---

🧠 Model Architecture

- Embedding Layer: Converts input word indices into dense vectors.
- SimpleRNN Layer: Single recurrent layer with 128 hidden units and ReLU activation.
- Dense Output Layer: Sigmoid activation for binary sentiment classification.

---

📁 Project Files

- simplernn.ipynb – Jupyter notebook for training the RNN
- model.h5 – Trained model weights
- tokenizer.json – Tokenizer used during preprocessing
- predict.py – Script to predict sentiment from a text input
- requirements.txt – List of dependencies

---

🚀 How to Use

1. Install Requirements

   pip install -r requirements.txt

2. Try the Web App

   Visit: https://rnn-movie-review-sentiment-analysis.streamlit.app/

---

📊 Dataset

- Source: IMDb Movie Reviews Dataset
- Preprocessed: Used via tensorflow.keras.datasets.imdb API

---

🛠️ Technologies Used

- Python, TensorFlow, Keras
- SimpleRNN layer
- Streamlit for the web interface

---

✨ Built for learning and experimentation in NLP & Deep Learning.
