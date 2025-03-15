
# Quora Duplicate Question Detection using LSTM and NLP


## Overview

This project focuses on detecting duplicate questions on Quora using Long Short-Term Memory (LSTM) networks and Natural Language Processing (NLP) techniques. The goal is to identify whether two questions are semantically similar, which can help in reducing redundancy and improving the quality of content on platforms like Quora.
## Dataset

The dataset used in this project consists of approximately 400,000 question pairs from Quora. Out of these, 200,000 question pairs were used for training and evaluation. Each pair is labeled as either duplicate (1) or non-duplicate (0).

Dataset Link - https://www.kaggle.com/c/quora-question-pairs
## Visualization


1.Bar Plot: Number of Duplicate vs Non-Duplicate Questions

2.Histogram: Unique vs Repeated Questions

## Preprocessing

1.Lowercasing: Convert all text to lowercase to ensure uniformity.


2.Special Character Replacement: Replace special characters with their string equivalents (e.g., $ with "dollar").

3.Number Replacement: Replace large numbers with their string equivalents (e.g., 1,000,000 with "1m").

4.HTML Tag Removal: Remove HTML tags using BeautifulSoup.

5.Punctuation Removal: Remove punctuation marks.

6.Stopwords: Common words like "the", "is", "and" are removed as they do not contribute much to the meaning.

7.Stemming: Words are reduced to their root form (e.g., "running" to "run") to ensure that different forms of the same word are treated equally.
## Features

The model utilizes 22 carefully engineered features to capture the similarity between question pairs. These features include:

1.Length-based Features

- q1_len: Length of question 1.

- q2_len: Length of question 2.

- abs_len_diff: Absolute difference in length between question 1 and question 2.

- mean_len: Mean length of the two questions.

2.Word-based Features

- q1_new_words: Number of unique words in question 1 that are not in question 2.

- q2_new_words: Number of unique words in question 2 that are not in question 1.

- word_common: Number of common words between the two questions.

- total_word: Total number of unique words in both questions.

- word_share: Ratio of common words to total words.

3.String Similarity Features

- longest_substr_ratio: Ratio of the length of the longest common substring to the length of the longer question.

- fuzzy_ratio: Fuzzy string matching ratio between the two questions.

- fuzzy_partial_ratio: Partial fuzzy string matching ratio.

- token_sort_ratio: Fuzzy string matching ratio after sorting tokens.

- token_set_ratio: Fuzzy string matching ratio after tokenizing and set operations.

4.Token-based Features

- cwc_min: Minimum common word count ratio.

- cwc_max: Maximum common word count ratio.

- csc_min: Minimum common stopword count ratio.

- csc_max: Maximum common stopword count ratio.

- ctc_min: Minimum common token count ratio.

- ctc_max: Maximum common token count ratio.

5.Position-based Features

- first_word_eq: Binary feature indicating if the first word of both questions is the same.

- last_word_eq: Binary feature indicating if the last word of both questions is the same.


## Tokenization and Padding

- Tokenization: Questions are tokenized to create a consistent word-to-index mapping.

- Padding: Each question is padded separately to maintain distinct sequences.


## Feature Normalization

- MinMaxScaler: Features are normalized to ensure that they are on the same scale, which helps in improving the performance of the model.




## Model Architecture

The model combines LSTM networks with additional feature inputs to predict whether two questions are duplicates. The architecture includes:

-Input Layers

- Question 1 Input: Input layer for the first question.

- Question 2 Input: Input layer for the second question.

- Features Input: Input layer for the additional 22 features.

-Embedding Layer

- Converts tokenized words into dense vectors.

-LSTM Layers

- LSTM for Question 1: Captures sequential information from the first question.

- LSTM for Question 2: Captures sequential information from the second question.

-Dense Layers

- Features Dense Layer: Processes additional features and combines them with LSTM outputs.

- Combined Dense Layer: Combines the outputs of the LSTM layers and the features dense layer.

-Output Layer

- Uses a sigmoid activation function to predict the probability of the questions being duplicates.

-Regularization
- Dropout: Used to prevent overfitting.

- Batch Normalization: Used to stabilize and speed up training.

-Compilation
- Optimizer: Adam optimizer with a learning rate of 0.0005.

- Loss Function: Binary cross-entropy.

- Metrics: Accuracy.


## Training

The model was trained using the Adam optimizer with a learning rate of 0.0005. Early stopping and learning rate reduction on plateau were employed to prevent overfitting and improve generalization. The training process achieved a validation accuracy of approximately 79.8%.
## Results

The model's performance was evaluated on a validation set, with the following key metrics:

- Training Accuracy: ~81.7%

- Validation Accuracy: ~79.8%

- Training Loss: ~0.3800

- Validation Loss: ~0.4234

The model shows promising results in detecting duplicate questions, with room for further improvement through hyperparameter tuning and additional feature engineering.
## Model Saving & Deployment

Saving the Model:

- Final Trained Model: The trained LSTM model is saved using pickle to enable easy loading and reuse for future predictions.

- Tokenizer: The tokenizer used for text preprocessing is saved using pickle to maintain consistent word-to-index mappings for new input data.

- Scaler: The MinMaxScaler used for feature normalization is saved using pickle to ensure consistent scaling of features during inference.

Deployment using Streamlit:

- A simple Streamlit app was developed to allow users to input questions and predict whether they are duplicates or not.
## Future Work

- Experiment with pre-trained embeddings like GloVe or Word2Vec.

- Tune hyperparameters and try GRU or Transformer-based models.


- Implement attention mechanisms for better contextual understanding.

## Acknowledgments

- Kaggle: For providing the dataset used in this project.

- TensorFlow/Keras: For the deep learning framework that enabled the implementation of the LSTM model.

- Scikit-learn: For providing tools for feature preprocessing and normalization.

- NLTK: For the stopwords and stemming functionalities used in text preprocessing.

- Special thanks to CampusX
Youtube Channel:https://www.youtube.com/@campusx-official
