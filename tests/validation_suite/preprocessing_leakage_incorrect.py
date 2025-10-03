"""
Preprocessing Leakage Validation - INCORRECT Implementation

This file demonstrates the most common and critical data leakage pattern:
preprocessing operations applied to the entire dataset before train/test split.

Expected: Multiple preprocessing leakage patterns detected
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


def incorrect_preprocessing_workflow():
    """INCORRECT: Preprocessing before split - classic leakage"""
    
    # Generate sample data
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # INCORRECT: Fit preprocessor on entire dataset before splitting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # LEAKAGE! Learns from test set
    
    # Split after preprocessing - too late!
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Model trained on leaked data
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)  # Overly optimistic!
    
    return accuracy


def multiple_leakage_patterns():
    """INCORRECT: Multiple preprocessors all causing leakage"""
    
    # Sample data
    X_numeric = np.random.randn(1000, 5)
    X_categorical = np.random.randint(0, 3, (1000, 2))
    y = np.random.randint(0, 2, 1000)
    
    # INCORRECT: All preprocessing before split
    numeric_scaler = MinMaxScaler()
    X_numeric_scaled = numeric_scaler.fit_transform(X_numeric)  # LEAKAGE!
    
    categorical_encoder = LabelEncoder()
    X_categorical_encoded = categorical_encoder.fit_transform(X_categorical.ravel())  # LEAKAGE!
    
    # Combine features after leakage
    X_combined = np.concatenate([X_numeric_scaled, X_categorical_encoded.reshape(-1, 2)], axis=1)
    
    # Split after all preprocessing - too late!
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test


def incorrect_robust_scaling():
    """INCORRECT: RobustScaler leakage before split"""
    
    # Sample data with outliers
    X = np.random.randn(1000, 8)
    X[50:60, :] = X[50:60, :] * 10  # Add outliers
    y = np.random.randint(0, 2, 1000)
    
    # INCORRECT: RobustScaler learns quartiles from entire dataset
    robust_scaler = RobustScaler()
    X_robust_scaled = robust_scaler.fit_transform(X)  # LEAKAGE! Uses test set quartiles
    
    # Split after scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X_robust_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test


def incorrect_imputation_leakage():
    """INCORRECT: Imputation before split"""
    
    # Sample data with missing values
    X = np.random.randn(1000, 6)
    X[np.random.choice(1000, 100), np.random.choice(6, 20)] = np.nan
    y = np.random.randint(0, 2, 1000)
    
    # INCORRECT: Imputer learns means from entire dataset including test set
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)  # LEAKAGE! Uses test set means
    
    # Split after imputation
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test


def incorrect_feature_engineering_leakage():
    """INCORRECT: Feature engineering using global statistics"""
    
    # Sample data
    X = np.random.randn(1000, 5)
    y = np.random.randn(1000)
    
    # INCORRECT: Compute statistics on entire dataset
    global_mean = X.mean(axis=0)  # LEAKAGE! Includes test set
    global_std = X.std(axis=0)    # LEAKAGE! Includes test set
    
    # Apply feature engineering
    X_normalized = (X - global_mean) / global_std
    
    # Split after feature engineering
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test


def incorrect_scaling_then_split():
    """INCORRECT: Classic scaling before split pattern"""
    
    # Generate dataset
    X = np.random.randn(1000, 12)
    y = np.random.randint(0, 3, 1000)
    
    # INCORRECT: Scaling first
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)  # LEAKAGE! Learns from test set
    
    # INCORRECT: Another scaler on already leaked data
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X_standardized)  # More leakage!
    
    # Split happens last - damage already done
    X_train, X_test, y_train, y_test = train_test_split(
        X_minmax, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model performance will be overly optimistic
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    return test_accuracy


def incorrect_cross_validation_leakage():
    """INCORRECT: Preprocessing before cross-validation"""
    
    # Sample data
    X = np.random.randn(500, 8)
    y = np.random.randint(0, 2, 500)
    
    # INCORRECT: Preprocessing before CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # LEAKAGE! Uses data from all CV folds
    
    # Cross-validation on already leaked data
    model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=5)  # Overly optimistic scores
    
    return scores.mean()


def multiple_preprocessor_leakage():
    """INCORRECT: Chain of preprocessing operations before split"""
    
    # Complex dataset
    X = np.random.randn(1000, 15)
    X[np.random.choice(1000, 50), :] = np.nan  # Add missing values
    y = np.random.randint(0, 2, 1000)
    
    # INCORRECT: Chain of leaky operations
    
    # Step 1: Imputation leakage
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)  # LEAKAGE!
    
    # Step 2: Scaling leakage  
    standard_scaler = StandardScaler()
    X_standardized = standard_scaler.fit_transform(X_imputed)  # LEAKAGE!
    
    # Step 3: More scaling leakage
    robust_scaler = RobustScaler()
    X_robust = robust_scaler.fit_transform(X_standardized)  # LEAKAGE!
    
    # Step 4: Final scaling leakage
    minmax_scaler = MinMaxScaler()
    X_final = minmax_scaler.fit_transform(X_robust)  # LEAKAGE!
    
    # Finally split - but all damage is done
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test


def subtle_leakage_example():
    """INCORRECT: Subtle leakage through feature selection"""
    
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # High-dimensional dataset
    X = np.random.randn(800, 100)
    y = np.random.randint(0, 2, 800)
    
    # INCORRECT: Feature selection before split
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X, y)  # LEAKAGE! Uses test set for selection
    
    # INCORRECT: Scaling after feature selection but before split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)  # Additional leakage!
    
    # Split after feature selection and scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )
    
    return X_train, X_test


# Example of problematic ML workflow
if __name__ == "__main__":
    # All these functions contain critical preprocessing leakage
    accuracy1 = incorrect_preprocessing_workflow()
    X_train1, X_test1 = multiple_leakage_patterns()
    X_train2, X_test2 = incorrect_robust_scaling()
    X_train3, X_test3 = incorrect_imputation_leakage()
    X_train4, X_test4 = incorrect_feature_engineering_leakage()
    accuracy2 = incorrect_scaling_then_split()
    cv_score = incorrect_cross_validation_leakage()
    X_train5, X_test5 = multiple_preprocessor_leakage()
    X_train6, X_test6 = subtle_leakage_example()
    
    print(f"All preprocessing operations performed incorrectly - causing data leakage!")