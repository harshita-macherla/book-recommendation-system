# book-recommendation-system
Sentiment-driven book recommendation system using BERT, VADER, and a 3-model hybrid deep learning ensemble (CNN-LSTM-GRU, CNN-BiLSTM-GRU, LSTM-CNN) achieving 98%+ accuracy on Amazon book reviews.

# Book Recommendation System using Sentiment Analysis and Hybrid Deep Learning (3-Model Ensemble)

A sentiment-driven book recommendation system that classifies Amazon book reviews into Positive, Neutral, and Negative sentiments using an ensemble of three hybrid deep learning models.

---

## Overview

This project performs sentiment analysis on Amazon book reviews using VADER for polarity scoring and BERT for contextual embeddings. Three hybrid deep learning models are trained and combined using a weighted soft voting ensemble for sentiment classification.

---

## Models Used

| Model | Description |
|-------|-------------|
| CNN-LSTM-GRU | CNN for local features + LSTM + GRU for sequential learning |
| CNN-BiLSTM-GRU | CNN + Bidirectional LSTM + GRU |
| LSTM-CNN | LSTM for sequential + CNN for feature extraction |

---

## Dataset
- Source: Amazon Books Reviews (Kaggle)
- Size: 3,160 reviews (after cleaning)
- Balanced to: 4,716 samples via oversampling (1,572 per class)

---

## Methodology
1. Preprocessing - Lowercasing, URL/special character removal, stopword removal, lemmatization
2. Sentiment Scoring - VADER compound polarity scores to Positive / Neutral / Negative labels
3. Oversampling - Random oversampling to balance class distribution
4. Feature Extraction - BERT (bert-base-uncased) 768-dim embeddings (96 tokens)
5. Feature Engineering - Normalized VADER polarity concatenated with BERT embeddings (96, 769) input
6. Model Training - Three hybrid models trained with EarlyStopping
7. Ensemble - Weighted soft voting across all three models
8. Deployment - Streamlit web app exposed via Ngrok

---

## Tech Stack
- Languages: Python
- Deep Learning: TensorFlow/Keras, PyTorch
- NLP: HuggingFace Transformers (BERT), NLTK (VADER)
- ML: Scikit-learn
- Visualization: Matplotlib, Seaborn
- Deployment: Streamlit, Ngrok
- Environment: Kaggle (GPU: NVIDIA Tesla T4)

---

## Contributors
- Harshita M (127157011) - B.Tech CSE (CS & BT), SASTRA University
- Amritha V S (127156008) - B.Tech CSE (AI & DS), SASTRA University
- Sujitha B (127156159) - B.Tech CSE (AI & DS), SASTRA University

Supervisor: Dr. G. Pradeep, Associate Professor, School of Computing, SASTRA University

---

## Reference
Devika, P., & Milton, A. (2024). Book recommendation using sentiment analysis and ensembling hybrid deep learning models. Knowledge and Information Systems, 67, 1131-1168. https://doi.org/10.1007/s10115-024-02250-z
