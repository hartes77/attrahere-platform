"""
Preprocessing Leakage Validation - CORRECT Implementation

This file demonstrates proper order-of-operations for ML preprocessing.
All preprocessing operations are applied after train/test split to prevent leakage.

Expected: 0 preprocessing leakage patterns detected
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def correct_preprocessing_workflow():
    """Correct: All preprocessing after train/test split"""
    
    # Generate sample data
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # CORRECT: Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # CORRECT: Fit preprocessor only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Learn from train only
    X_test_scaled = scaler.transform(X_test)        # Apply to test without learning
    
    # Train and evaluate model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy


def correct_multiple_preprocessors():
    """Correct: Multiple preprocessors all applied after split"""
    
    # Sample data with mixed types
    X_numeric = np.random.randn(1000, 5)
    X_categorical = np.random.randint(0, 3, (1000, 2))
    y = np.random.randint(0, 2, 1000)
    
    # CORRECT: Split first
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_numeric, X_categorical, y, test_size=0.2, random_state=42
    )
    
    # CORRECT: Each preprocessor fitted only on training data
    numeric_scaler = StandardScaler()
    X_num_train_scaled = numeric_scaler.fit_transform(X_num_train)
    X_num_test_scaled = numeric_scaler.transform(X_num_test)
    
    categorical_encoder = LabelEncoder()
    X_cat_train_encoded = categorical_encoder.fit_transform(X_cat_train.ravel()).reshape(-1, 2)
    X_cat_test_encoded = categorical_encoder.transform(X_cat_test.ravel()).reshape(-1, 2)
    
    # Combine features
    X_train_processed = np.concatenate([X_num_train_scaled, X_cat_train_encoded], axis=1)
    X_test_processed = np.concatenate([X_num_test_scaled, X_cat_test_encoded], axis=1)
    
    return X_train_processed, X_test_processed


def correct_pipeline_approach():
    """Correct: Using sklearn Pipeline to ensure proper order"""
    
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    # Sample data
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # CORRECT: Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # CORRECT: Pipeline ensures preprocessing only fits on training data
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Fitted only on training data
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Pipeline automatically handles proper preprocessing order
    pipeline.fit(X_train, y_train)         # Scaler learns from train only
    predictions = pipeline.predict(X_test) # Scaler transforms test without learning
    
    return predictions


def correct_cross_validation():
    """Correct: Cross-validation with proper preprocessing order"""
    
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    
    # Sample data
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # CORRECT: Pipeline ensures preprocessing is done within each CV fold
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # CORRECT: cross_val_score applies preprocessing separately for each fold
    scores = cross_val_score(pipeline, X, y, cv=5)
    
    return scores.mean()


def correct_feature_engineering():
    """Correct: Feature engineering after split"""
    
    # Sample time series data
    X = np.random.randn(1000, 5)
    y = np.random.randn(1000)
    
    # CORRECT: Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # CORRECT: Feature engineering statistics computed only on training data
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    
    # Apply same transformations to both sets
    X_train_normalized = (X_train - train_mean) / train_std
    X_test_normalized = (X_test - train_mean) / train_std    # Use training stats
    
    return X_train_normalized, X_test_normalized


def correct_scaling_workflow():
    """Correct: Complete workflow with proper scaling order"""
    
    # Sample data
    X = np.random.randn(1000, 8)
    y = np.random.randint(0, 3, 1000)
    
    # Step 1: CORRECT - Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 2: CORRECT - Initialize preprocessors
    scaler = MinMaxScaler()
    
    # Step 3: CORRECT - Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train)  # Learn scaling parameters from training only
    
    # Step 4: CORRECT - Transform test data using training parameters
    X_test_scaled = scaler.transform(X_test)        # Apply scaling without learning
    
    # Step 5: Train model on properly preprocessed data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Step 6: Evaluate on properly preprocessed test data
    test_accuracy = model.score(X_test_scaled, y_test)
    
    return test_accuracy


def correct_imputation_workflow():
    """Correct: Missing value imputation after split"""
    
    from sklearn.impute import SimpleImputer
    
    # Sample data with missing values
    X = np.random.randn(1000, 6)
    X[np.random.choice(1000, 100), np.random.choice(6, 20)] = np.nan  # Add missing values
    y = np.random.randint(0, 2, 1000)
    
    # CORRECT: Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # CORRECT: Imputer learns strategy only from training data
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)  # Learn mean from training only
    X_test_imputed = imputer.transform(X_test)        # Apply same means to test
    
    return X_train_imputed, X_test_imputed


# Example of proper ML workflow
if __name__ == "__main__":
    # All these functions follow correct preprocessing order
    accuracy1 = correct_preprocessing_workflow()
    X_train, X_test = correct_multiple_preprocessors()
    predictions = correct_pipeline_approach()
    cv_score = correct_cross_validation()
    X_train_norm, X_test_norm = correct_feature_engineering()
    test_acc = correct_scaling_workflow()
    X_train_imp, X_test_imp = correct_imputation_workflow()
    
    print(f"All preprocessing operations performed correctly after data splitting")