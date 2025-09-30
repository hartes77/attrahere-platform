# Test case for inefficient data loading - PROBLEMATIC CODE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ❌ INEFFICIENT: Loading entire dataset into memory without chunking
df = pd.read_csv('huge_dataset.csv')  # Could be 10GB+ file

# ❌ INEFFICIENT: Reading all files at once instead of lazy loading
import os
all_files = []
for file in os.listdir('data/'):
    if file.endswith('.csv'):
        all_files.append(pd.read_csv(f'data/{file}'))  # Memory explosion
df_combined = pd.concat(all_files)

# ❌ INEFFICIENT: No chunking for large data processing
def process_data():
    # Loading everything in memory
    for i in range(len(df)):  # Iterating row by row instead of vectorizing
        row = df.iloc[i]  # Very slow indexing
        # Process each row individually
        result = row['value'] * 2
        df.loc[i, 'processed'] = result

# ❌ INEFFICIENT: Multiple redundant data loads
df1 = pd.read_csv('data.csv')  # First load
df2 = pd.read_csv('data.csv')  # Same file loaded again!
df3 = pd.read_csv('data.csv')  # And again!

# ❌ INEFFICIENT: Loading data without specifying dtypes (memory waste)
df_untyped = pd.read_csv('numeric_data.csv')  # All columns become object/float64

# ❌ INEFFICIENT: Loading all columns when only few are needed
df_all_columns = pd.read_csv('wide_dataset.csv')  # 500 columns but using only 3
X = df_all_columns[['feature1', 'feature2', 'feature3']]

# ❌ INEFFICIENT: No data streaming for large datasets
def train_model():
    # Loading everything at once
    X_train, X_test, y_train, y_test = train_test_split(df_combined.drop('target', axis=1), 
                                                        df_combined['target'])
    # This could cause memory errors on large datasets