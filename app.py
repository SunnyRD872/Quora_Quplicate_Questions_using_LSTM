import streamlit as st
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fuzzywuzzy import fuzz
from tensorflow.keras.models import load_model

# Load trained model and scaler
@st.cache_resource
def load_resources():
    with open('quora_duplicate_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    return model, scaler, tokenizer

model, scaler, tokenizer = load_resources()

# Constants
VOCAB_SIZE = 5000
MAX_LEN = 20

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.replace('?', ' ')
    text = text.replace('.', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# Feature extraction function
def extract_features_v2(q1, q2):
    q1_clean = clean_text(q1)
    q2_clean = clean_text(q2)

    q1_words = set(q1_clean.split())
    q2_words = set(q2_clean.split())

    q1_len = len(q1_clean.split())
    q2_len = len(q2_clean.split())
    abs_len_diff = abs(q1_len - q2_len)
    mean_len = (q1_len + q2_len) / 2

    q1_new_words = len(q1_words - q2_words)
    q2_new_words = len(q2_words - q1_words)

    word_common = len(q1_words & q2_words)
    total_word = len(q1_words | q2_words)
    word_share = word_common / total_word if total_word > 0 else 0

    fuzzy_ratio = fuzz.ratio(q1_clean, q2_clean)
    fuzzy_partial_ratio = fuzz.partial_ratio(q1_clean, q2_clean)
    token_sort_ratio = fuzz.token_sort_ratio(q1_clean, q2_clean)
    token_set_ratio = fuzz.token_set_ratio(q1_clean, q2_clean)

    def longest_substring_ratio(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        longest = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    longest = max(longest, dp[i][j])
        return longest / max(m, n) if max(m, n) > 0 else 0

    longest_substr_ratio = longest_substring_ratio(q1_clean, q2_clean)

    last_word_eq = int(q1_clean.split()[-1] == q2_clean.split()[-1]) if q1_clean and q2_clean else 0
    first_word_eq = int(q1_clean.split()[0] == q2_clean.split()[0]) if q1_clean and q2_clean else 0

    def count_word_matches(w1, w2):
        w1_set = set(w1)
        w2_set = set(w2)
        return len(w1_set & w2_set) / min(len(w1_set), len(w2_set)) if min(len(w1_set), len(w2_set)) > 0 else 0

    cwc_min = count_word_matches(q1_words, q2_words)
    cwc_max = len(q1_words & q2_words) / max(len(q1_words), len(q2_words)) if max(len(q1_words), len(q2_words)) > 0 else 0

    csc_min = count_word_matches(q1_clean.split(), q2_clean.split())
    csc_max = len(q1_clean.split()) / max(len(q1_clean.split()), len(q2_clean.split())) if max(len(q1_clean.split()), len(q2_clean.split())) > 0 else 0

    ctc_min = count_word_matches(set(q1_clean), set(q2_clean))
    ctc_max = len(set(q1_clean) & set(q2_clean)) / max(len(set(q1_clean)), len(set(q2_clean))) if max(len(set(q1_clean)), len(set(q2_clean))) > 0 else 0

    features = [
        q1_len, q2_len, q1_new_words, q2_new_words, 
        word_common, total_word, word_share, 
        abs_len_diff, mean_len, longest_substr_ratio, 
        fuzzy_ratio, fuzzy_partial_ratio, token_sort_ratio, token_set_ratio, 
        cwc_min, cwc_max, csc_min, csc_max,
        ctc_min, ctc_max, last_word_eq, first_word_eq
    ]
    return np.array(features).reshape(1, -1)

# Streamlit UI
st.title("Quora Duplicate Question Pair Detector")
st.write("Enter two questions to check if they are duplicates.")

q1_input = st.text_input("Enter Question 1")
q2_input = st.text_input("Enter Question 2")

if st.button("Check Similarity"):
    if q1_input and q2_input:
        q1_clean = clean_text(q1_input)
        q2_clean = clean_text(q2_input)

        features = extract_features_v2(q1_clean, q2_clean)
        features_scaled = scaler.transform(features)

        q1_seq = tokenizer.texts_to_sequences([q1_clean])
        q2_seq = tokenizer.texts_to_sequences([q2_clean])

        q1_padded = pad_sequences(q1_seq, maxlen=MAX_LEN, padding='post')
        q2_padded = pad_sequences(q2_seq, maxlen=MAX_LEN, padding='post')

        prediction = model.predict([q1_padded, q2_padded, features_scaled])

        result = "Duplicate" if prediction[0][0] > 0.5 else "Not Duplicate"
        st.subheader(f"Result: {result}")
        st.write(f"Confidence: {prediction[0][0]:.2f}")
    else:
        st.warning("Please enter both questions.")
