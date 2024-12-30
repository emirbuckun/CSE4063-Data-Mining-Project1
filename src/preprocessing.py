from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from scipy.sparse import hstack

def load_and_explore_data(X, y):
    """Load and explore the dataset"""
    print("\n=== Dataset Analysis ===")
    print(f"Dataset size: {X.shape}")
    print("\nFeatures:")
    print(X.info())
    print("\nMissing values:")
    print(X.isnull().sum())
    print("\nTarget variable (stars) distribution:")
    print(y.value_counts().sort_index())
    print("\nTarget variable statistics:")
    print(y.describe())
    return X, y

def preprocess_data(X, y):
    """Data preprocessing"""
    # Remove rows with 0 stars
    non_zero_indices = y != 0
    removed_rows = X.shape[0] - non_zero_indices.sum()
    print(f"Rows removed after filtering non-zero stars: {removed_rows}")
    X = X[non_zero_indices]
    y = y[non_zero_indices]
    
    # Fill missing values
    X = X.copy()
    X['text'] = X['text'].fillna('')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Process text and numeric data separately
    # TF-IDF transformation for text data
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_text = tfidf.fit_transform(X_train['text'])
    X_test_text = tfidf.transform(X_test['text'])
    
    # MinMaxScaler for numeric features (normalize between 0-1)
    numeric_features = ['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'best_score']
    scaler = MinMaxScaler()
    
    # Convert numeric features to numpy array
    X_train_num = scaler.fit_transform(X_train[numeric_features].values)
    X_test_num = scaler.transform(X_test[numeric_features].values)
    
    # Convert to sparse matrix
    X_train_num_sparse = sparse.csr_matrix(X_train_num)
    X_test_num_sparse = sparse.csr_matrix(X_test_num)
    
    # Combine text and numeric features
    X_train_processed = hstack([X_train_text, X_train_num_sparse])
    X_test_processed = hstack([X_test_text, X_test_num_sparse])
    
    # Non-negative check
    print("\nData Check:")
    print(f"Any negative values (Train): {(X_train_processed.data < 0).any()}")
    print(f"Any negative values (Test): {(X_test_processed.data < 0).any()}")
    print(f"Minimum value (Train): {X_train_processed.data.min():.6f}")
    print(f"Maximum value (Train): {X_train_processed.data.max():.6f}")
    
    preprocessor = {
        'tfidf': tfidf,
        'scaler': scaler,
        'numeric_features': numeric_features
    }
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor