
import streamlit as st
import numpy as np
import torch
import pickle
import tensorflow as tf
from transformers import BertTokenizer, BertModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import nltk, warnings, datetime

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="Book Review Analyzer", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextArea textarea { background-color: #1e2130; color: #e0e0e0;
        border: 1px solid #4a90d9; border-radius: 10px; font-size: 15px; }
    .result-box { padding: 20px; border-radius: 12px; margin: 15px 0; }
    .metric-card { background: #1e2130; border-radius: 10px; padding: 15px;
        text-align: center; border: 1px solid #2e3250; }
    .history-item { background: #1e2130; border-left: 4px solid #4a90d9;
        padding: 10px 15px; margin: 8px 0; border-radius: 0 8px 8px 0; font-size: 13px; }
    h1 { color: #e8f4fd !important; }
    .stButton>button { border-radius: 8px; font-weight: 600; transition: all 0.2s; }
</style>
""", unsafe_allow_html=True)

st.title("Book Review Sentiment Analyzer")
st.caption("4-Model Ensemble: CNN-LSTM-GRU · CNN-BiLSTM-GRU · LSTM-CNN · BiLSTM-CNN — BERT + VADER features")

if "history" not in st.session_state: st.session_state.history = []
if "review_text" not in st.session_state: st.session_state.review_text = ""
if "result" not in st.session_state: st.session_state.result = None

@st.cache_resource
def load_everything():
    m1 = tf.keras.models.load_model("/kaggle/working/cnn_lstm_gru_final.keras")
    m2 = tf.keras.models.load_model("/kaggle/working/cnn_bilstm_gru_final.keras")
    m3 = tf.keras.models.load_model("/kaggle/working/lstm_cnn_final.keras")
    m4 = tf.keras.models.load_model("/kaggle/working/bilstm_cnn_final.keras")
    with open("/kaggle/working/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device).eval()
    return m1, m2, m3, m4, scaler, tokenizer, bert_model, device

m1, m2, m3, m4, scaler, tokenizer, bert_model, device = load_everything()

vader = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

LABELS = ["Negative", "Neutral", "Positive"]
COLORS = ["#FF4B4B", "#FFA500", "#00C853"]
MODEL_NAMES = ["CNN-LSTM-GRU", "CNN-BiLSTM-GRU", "LSTM-CNN", "BiLSTM-CNN"]
W = [0.20, 0.35, 0.15, 0.30]

def get_bert_embedding(text):
    enc = tokenizer([text], padding="max_length", truncation=True,
                    max_length=96, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = bert_model(**enc)
    embedding = out.last_hidden_state.cpu().numpy()
    return embedding

def predict(review_text):
    # VADER polarity
    polarity = vader.polarity_scores(str(review_text))["compound"]
    norm_polarity = float(scaler.transform([[polarity]])[0][0])
    
    # BERT embedding (1, 96, 768)
    X_emb = get_bert_embedding(review_text)
    
    # Tile polarity (1, 96, 1)
    pol_tile = np.tile(np.array([[[norm_polarity]]]), (1, X_emb.shape[1], 1))
    
    # Concatenate (1, 96, 769)
    X = np.concatenate([X_emb, pol_tile], axis=-1)
    
    # Predict with all 4 models
    p1 = m1.predict(X, verbose=0)
    p2 = m2.predict(X, verbose=0)
    p3 = m3.predict(X, verbose=0)
    p4 = m4.predict(X, verbose=0)
    
    # Weighted ensemble
    ensemble = W[0]*p1 + W[1]*p2 + W[2]*p3 + W[3]*p4
    pred_idx = int(np.argmax(ensemble))
    confidence = float(ensemble[0][pred_idx]) * 100
    
    return pred_idx, confidence, ensemble[0], p1[0], p2[0], p3[0], p4[0]

with st.sidebar:
    st.markdown("### About")
    st.info("4-model deep learning ensemble trained on Amazon Books Reviews. Features: BERT (96-token) + VADER polarity via StandardScaler.")
    
    st.markdown("### Ensemble Weights")
    for name, w in zip(MODEL_NAMES, W):
        st.markdown(f"""<div class="metric-card" style="margin-bottom:8px">
            <b style="color:#4a90d9">{name}</b><br>
            <span style="font-size:20px;color:#00C853"><b>{int(w*100)}%</b></span>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("### History")
    if st.session_state.history:
        if st.button("Clear History"): 
            st.session_state.history = []; 
            st.rerun()
        for item in reversed(st.session_state.history[-10:]):
            color = COLORS[item["pred_idx"]]
            st.markdown(f"""<div class="history-item">
                <b style="color:{color}">{LABELS[item["pred_idx"]]}</b> · {item["conf"]:.0f}%<br>
                <span style="color:#888">{item["time"]}</span><br>
                <span style="color:#aaa;font-size:12px">"{item["review"][:60]}..."</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("No history yet.")

st.markdown("**Try an example:**")
ex1, ex2, ex3 = st.columns(3)
if ex1.button("Positive", use_container_width=True):
    st.session_state.review_text = "This book was absolutely brilliant! The characters felt so real and the plot kept me hooked till the very last page. Highly recommended!"
    st.rerun()
if ex2.button("Neutral", use_container_width=True):
    st.session_state.review_text = "The book was okay, not much to read, just an average story with nothing special to remember."
    st.rerun()
if ex3.button("Negative", use_container_width=True):
    st.session_state.review_text = "I was really disappointed. The story was boring, characters were flat, and the ending made no sense. Not worth the time."
    st.rerun()

col_input, col_result = st.columns([1.1, 1], gap="large")

with col_input:
    st.markdown("#### Enter Book Review")
    review = st.text_area("", height=180, placeholder="e.g. This book was absolutely wonderful...",
        label_visibility="collapsed", value=st.session_state.review_text)
    
    c1, c2 = st.columns(2)
    analyze_btn = c1.button("Analyze Sentiment", use_container_width=True, type="primary")
    clear_btn = c2.button("Clear Text", use_container_width=True)
    
    if clear_btn:
        st.session_state.review_text = ""
        st.session_state.result = None
        st.rerun()

with col_result:
    st.markdown("#### Results")
    if analyze_btn:
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            st.session_state.review_text = review
            with st.spinner("Running BERT + 4-Model Ensemble..."):
                idx, conf, ens, p1, p2, p3, p4 = predict(review)
            
            st.session_state.result = (idx, conf, ens, p1, p2, p3, p4)
            st.session_state.history.append({
                "pred_idx": idx, 
                "conf": conf,
                "review": review.strip(), 
                "time": datetime.datetime.now().strftime("%H:%M:%S")
            })

    if st.session_state.result:
        idx, conf, ens, p1, p2, p3, p4 = st.session_state.result
        label = LABELS[idx]
        color = COLORS[idx]
        
        st.markdown(f"""<div class="result-box" style="background:{color}22; border-left:5px solid {color}">
            <h2 style="margin:0;color:{color}">{label}</h2>
            <p style="margin:5px 0 0 0;color:#ccc">Ensemble Confidence: <b style="color:white">{conf:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
        
        st.progress(conf / 100)
        
        st.markdown("##### All 4 Model Predictions")
        model_df = pd.DataFrame({
            "Model": MODEL_NAMES,
            "Weight": [f"{int(w*100)}%" for w in W],
            "Negative %": [p1[0]*100, p2[0]*100, p3[0]*100, p4[0]*100],
            "Neutral %": [p1[1]*100, p2[1]*100, p3[1]*100, p4[1]*100],
            "Positive %": [p1[2]*100, p2[2]*100, p3[2]*100, p4[2]*100],
        }).set_index("Model")
        
        st.dataframe(model_df.style.format({"Negative %":"{:.1f}","Neutral %":"{:.1f}","Positive %":"{:.1f}"}),
                     use_container_width=True)
        
        st.markdown("##### Weighted Ensemble Score")
        st.bar_chart(pd.DataFrame({"Score": list(ens)}, index=["Negative","Neutral","Positive"]))
        
        polarity = vader.polarity_scores(str(st.session_state.review_text))["compound"]
        m1c, m2c = st.columns(2)
        m1c.metric("VADER Polarity", f"{polarity:.3f}")
        m2c.metric("Predicted Label", label)
        
    else:
        st.info("Results will appear here after analysis.")
