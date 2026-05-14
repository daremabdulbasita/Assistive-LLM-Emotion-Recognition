"""
Emotion Recognition in Disabled Individuals Through an Improved Assistive Communication System Based on Large Language Models and Ensemble Deep Learning
Official Implementation for 'The Visual Computer'

This script implements the TERLLMV-DLN architecture, which combines TCN, SDAE, 
and ENN modules for high-accuracy emotion classification from text.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, matthews_corrcoef
)

from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, BatchNormalization,
    GlobalAveragePooling1D, SimpleRNN, concatenate, GaussianNoise,
    Bidirectional, LSTM
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ============================================================
# CONFIGURATION
# ============================================================
DATASET1_PATH = 'emotions_dataset1.csv'
DATASET2_PATH = 'emotions_dataset2.csv'
MIN_SAMPLES = 10000
MODEL_NAME = 'all-MiniLM-L6-v2'

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = str(text).lower()
    text = re.sub(r'http\S+|@\w+|#\w+|\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def detect_columns(df):
    text_col = next((c for c in df.columns if any(k in c.lower() for k in ['text', 'sentence', 'content'])), None)
    label_col = next((c for c in df.columns if any(k in c.lower() for k in ['label', 'emotion', 'sentiment'])), None)
    return text_col, label_col

# ============================================================
# MODEL DEFINITION (TERLLMV-DLN)
# ============================================================
def build_terllmv_dln(input_shape, num_classes):
    # TCN Branch
    input_tcn = Input(shape=(input_shape, 1), name='TCN_Input')
    x1 = Conv1D(64, 3, dilation_rate=1, padding='causal', activation='relu')(input_tcn)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(128, 3, dilation_rate=2, padding='causal', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(256, 3, dilation_rate=4, padding='causal', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalAveragePooling1D()(x1)

    # SDAE Branch
    input_sdae = Input(shape=(input_shape,), name='SDAE_Input')
    x2 = GaussianNoise(0.1)(input_sdae)
    x2 = Dense(512, activation='relu')(x2)
    x2 = Dropout(0.3)(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(0.3)(x2)
    x2 = Dense(128, activation='relu')(x2)

    # ENN Branch
    input_enn = Input(shape=(input_shape, 1), name='ENN_Input')
    x3 = SimpleRNN(128, return_sequences=True)(input_enn)
    x3 = SimpleRNN(64)(x3)

    # Fusion
    merged = concatenate([x1, x2, x3])
    fc = Dense(256, activation='relu')(merged)
    fc = Dropout(0.4)(fc)
    fc = Dense(128, activation='relu')(fc)
    fc = Dropout(0.3)(fc)
    output = Dense(num_classes, activation='softmax')(fc)

    model = Model(inputs=[input_tcn, input_sdae, input_enn], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================
if __name__ == "__main__":
    # Load and Preprocess (Simplified for script demonstration)
    # In a real run, ensure datasets are present
    try:
        print("Loading datasets...")
        df1 = pd.read_csv(DATASET1_PATH)
        text_col, label_col = detect_columns(df1)
        
        print(f"Using column '{text_col}' for text and '{label_col}' for labels.")
        
        # Balanced sampling
        df = df1.groupby(label_col).apply(lambda x: x.sample(min(len(x), MIN_SAMPLES))).reset_index(drop=True)
        
        print("Preprocessing text...")
        df['clean_text'] = df[text_col].apply(preprocess_text)
        
        le = LabelEncoder()
        y = le.fit_transform(df[label_col])
        num_classes = len(le.classes_)
        
        print("Generating LLM Embeddings (Sentence Transformers)...")
        embed_model = SentenceTransformer(MODEL_NAME)
        X = embed_model.encode(df['clean_text'].tolist(), show_progress_bar=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        
        # Reshaping for TCN/ENN
        X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_train_cat = to_categorical(y_train, num_classes)
        
        print("Building and training TERLLMV-DLN...")
        model = build_terllmv_dln(X.shape[1], num_classes)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit([X_train_3d, X_train, X_train_3d], y_train_cat, 
                  validation_split=0.2, epochs=25, batch_size=64, callbacks=[early_stop])
        
        # Save model
        model.save('TERLLMV_DLN_Model.h5')
        print("Model training complete and saved.")
        
    except FileNotFoundError:
        print(f"Error: Datasets not found at {DATASET1_PATH}. Please download them first.")
