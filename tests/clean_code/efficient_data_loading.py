# Test case for efficient data loading - CLEAN CODE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ✅ EFFICIENT: Chunked data loading for large files
def load_data_efficiently(file_path, chunk_size=10000):
    """Load large CSV files in chunks to manage memory usage."""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process each chunk before loading next
        chunk_processed = chunk.dropna()  # Example processing
        chunks.append(chunk_processed)
    return pd.concat(chunks, ignore_index=True)

# ✅ EFFICIENT: Specify data types to reduce memory usage
dtype_spec = {
    'user_id': 'int32',          # Instead of int64
    'score': 'float32',          # Instead of float64
    'category': 'category',      # Instead of object
    'is_active': 'bool'          # Boolean instead of object
}
df_optimized = pd.read_csv('data.csv', dtype=dtype_spec)

# ✅ EFFICIENT: Load only required columns
required_columns = ['feature1', 'feature2', 'feature3', 'target']
df_selective = pd.read_csv('wide_dataset.csv', usecols=required_columns)

# ✅ EFFICIENT: Lazy loading with generators for multiple files
def load_files_generator(data_dir):
    """Generator for lazy loading of multiple files."""
    import os
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            yield pd.read_csv(os.path.join(data_dir, filename))

# ✅ EFFICIENT: Process data using vectorized operations
def process_data_vectorized(df):
    """Efficient vectorized data processing."""
    # Vectorized operation instead of row-by-row iteration
    df['processed'] = df['value'] * 2
    
    # Use built-in pandas methods for efficiency
    df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
    
    return df

# ✅ EFFICIENT: Data caching to avoid redundant loads
import functools

@functools.lru_cache(maxsize=None)
def load_cached_data(file_path):
    """Cache loaded data to avoid redundant file reads."""
    return pd.read_csv(file_path)

# Use cached loading
df1 = load_cached_data('data.csv')  # Loaded from file
df2 = load_cached_data('data.csv')  # Loaded from cache (same object)

# ✅ EFFICIENT: Streaming data processing for ML training
def train_model_streaming(data_generator, model):
    """Train model using data streaming to handle large datasets."""
    for batch in data_generator:
        X_batch = batch.drop('target', axis=1)
        y_batch = batch['target']
        
        # Partial fit for incremental learning
        model.partial_fit(X_batch, y_batch)
    
    return model

# ✅ EFFICIENT: Memory-conscious data preprocessing
def preprocess_efficiently(df):
    """Efficient preprocessing with memory management."""
    # Process in-place to save memory
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Use categorical encoding for memory efficiency
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() < 50:  # Only if reasonable number of categories
            df[col] = df[col].astype('category')
    
    return df

# ✅ EFFICIENT: Example usage
if __name__ == "__main__":
    # Load data efficiently
    df = load_data_efficiently('large_dataset.csv')
    
    # Preprocess efficiently
    df = preprocess_efficiently(df)
    
    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)