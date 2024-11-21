import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse  # Bu satırı ekledik
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(X, y):
    """Veri setini yükle ve keşifsel analiz yap"""
    print("\n=== Veri Seti Analizi ===")
    print(f"Veri seti boyutu: {X.shape}")
    print("\nÖzellikler:")
    print(X.info())
    print("\nEksik değerler:")
    print(X.isnull().sum())
    print("\nHedef değişken (yıldız) dağılımı:")
    print(y.value_counts().sort_index())
    print("\nHedef değişken istatistikleri:")
    print(y.describe())
    return X, y

def preprocess_data(X, y):
    """Veri ön işleme"""
    # Eksik değerleri doldur
    X = X.copy()
    X['text'] = X['text'].fillna('')
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Metin verilerini ve sayısal verileri ayrı ayrı işle
    # TF-IDF için metin dönüşümü
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_text = tfidf.fit_transform(X_train['text'])
    X_test_text = tfidf.transform(X_test['text'])
    
    # Sayısal özellikler için MinMaxScaler (0-1 arasına normalize et)
    numeric_features = ['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'best_score']
    scaler = MinMaxScaler()  # StandardScaler yerine MinMaxScaler kullan
    
    # Sayısal özellikleri numpy array'e dönüştür
    X_train_num = scaler.fit_transform(X_train[numeric_features].values)
    X_test_num = scaler.transform(X_test[numeric_features].values)
    
    # Sparse matrise dönüştür
    X_train_num_sparse = sparse.csr_matrix(X_train_num)
    X_test_num_sparse = sparse.csr_matrix(X_test_num)
    
    # Metin ve sayısal özellikleri birleştir
    X_train_processed = hstack([X_train_text, X_train_num_sparse])
    X_test_processed = hstack([X_test_text, X_test_num_sparse])
    
    # Non-negative kontrolü
    print("\nVeri Kontrol:")
    print(f"Negatif değer var mı (Train): {(X_train_processed.data < 0).any()}")
    print(f"Negatif değer var mı (Test): {(X_test_processed.data < 0).any()}")
    print(f"Minimum değer (Train): {X_train_processed.data.min():.6f}")
    print(f"Maximum değer (Train): {X_train_processed.data.max():.6f}")
    
    preprocessor = {
        'tfidf': tfidf,
        'scaler': scaler,
        'numeric_features': numeric_features
    }
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor