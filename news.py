import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("model.h5")
tokenizer = joblib.load("tokenizer.joblib")
encoder = joblib.load("encoder.joblib")

st.title("📰 Klasifikasi Berita Otomatis")
st.markdown("Masukkan judul atau isi berita, dan model akan menebak kategorinya!")

# Input user
text_input = st.text_area("Masukkan teks berita di sini:", height=200)

def clean_text(text):
    text = text.lower()
    text = text.replace(",", "")
    text = text.replace('"', "")
    text = text.replace("-", "")
    text = text.replace(".", "")
    text = text.replace(":", "")
    text = text.replace(")", "")
    text = text.replace("(", "")
    text = text.replace("/", "")
    text = text.replace("\n", "")
    return text

if st.button("Klasifikasi Berita"):
    if text_input.strip() == "":
        st.warning("Tolong isi teks berita terlebih dahulu.")
    else:
        input_data = {
            "content": [
                text_input
            ]
        }
        input_data_df = pd.DataFrame(input_data)
        input_data_df['content'] = input_data_df['content'].apply(clean_text)
        input_data_seq = tokenizer.texts_to_sequences(input_data_df['content'])
        input_data_pad = pad_sequences(input_data_seq, padding="post", maxlen=50)
        prediction = model.predict(input_data_pad)
        prediction_classes = np.argmax(prediction, axis=1)
        prediction_encoded = encoder.inverse_transform(prediction_classes)
        st.success("""Hasil Klasifikasi: {}""".format(prediction_encoded[0]))