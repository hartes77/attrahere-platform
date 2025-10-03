"""
CLEAN CODE - Test Set Contamination Prevention
EXPECTED ERRORS: 0

Questo file dimostra il modo CORRETTO di evitare contaminazioni del test set:
1. Duplicati: split temporale corretto per time series
2. Feature leakage: rimozione di features con informazioni future
3. Temporal leakage: split chronologico in dati temporali
4. Preprocessing leakage: fit solo su training data
5. Target leakage: rimozione di features correlate al target

Questi pattern complessi devono essere riconosciuti come PULITI dal detector.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CleanContaminationPrevention:
    """Esempi di operazioni complesse ma pulite per evitare contaminazione"""
    
    def __init__(self):
        """Setup con seeds per reproducibilità"""
        np.random.seed(42)
        random.seed(42)
    
    def clean_temporal_split_example(self):
        """
        ✅ CORRETTO: Split temporale per time series
        Dimostra come gestire correttamente dati temporali senza contaminazione
        """
        # Genera time series data realistici
        n_samples = 2000
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]
        
        # Features che evolvono nel tempo
        trend = np.linspace(0, 10, n_samples)
        seasonality = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        noise = np.random.normal(0, 1, n_samples)
        
        # Target che dipende da trend passato (legittimo)
        y = (trend + seasonality + noise > 5).astype(int)
        
        # Features legittime (non future information)
        X = pd.DataFrame({
            'trend_lag1': np.roll(trend, 1),  # Lag di 1 giorno - OK
            'trend_lag7': np.roll(trend, 7),  # Lag di 7 giorni - OK  
            'seasonality_lag1': np.roll(seasonality, 1),  # Seasonality passata - OK
            'noise_lag1': np.roll(noise, 1),  # Noise passato - OK
            'moving_avg_30d': pd.Series(trend).rolling(30, min_periods=1).mean(),  # Media mobile - OK
            'day_of_year': [d.timetuple().tm_yday for d in dates],  # Giorno dell'anno - OK
            'month': [d.month for d in dates],  # Mese - OK
        })
        
        # Rimuovi prime righe con valori NaN dai lag
        X = X.iloc[30:]  # Remove first 30 days per rolling window
        y = y[30:]
        dates = dates[30:]
        
        # ✅ CORRETTO: Split temporale (70% train, 30% test)
        split_point = int(0.7 * len(X))
        
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        
        # ✅ CORRETTO: Preprocessing solo su training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ✅ CORRETTO: Training con dati puliti
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Valutazione
        train_score = accuracy_score(y_train, model.predict(X_train_scaled))
        test_score = accuracy_score(y_test, model.predict(X_test_scaled))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'temporal_split': True,
            'no_future_leakage': True
        }
    
    def clean_feature_engineering_example(self):
        """
        ✅ CORRETTO: Feature engineering senza target leakage
        Dimostra creazione di features complesse ma legittime
        """
        # Dataset simulato per e-commerce
        n_customers = 1000
        
        # Features demografiche (disponibili al momento della predizione)
        ages = np.random.randint(18, 80, n_customers)
        incomes = np.random.lognormal(10, 0.5, n_customers)
        regions = np.random.choice(['North', 'South', 'East', 'West'], n_customers)
        
        # Storico acquisti (informazioni passate - legittime)
        past_purchases = np.random.poisson(5, n_customers)
        avg_order_value = np.random.lognormal(4, 0.3, n_customers)
        days_since_last_purchase = np.random.exponential(30, n_customers)
        
        # Target: churn in prossimi 30 giorni (simulato)
        churn_probability = (
            0.1 + 
            0.001 * ages +  # Older customers slightly more likely to churn
            -0.00001 * incomes +  # Higher income less likely to churn
            -0.01 * past_purchases +  # More purchases = less churn
            0.001 * days_since_last_purchase  # More days since purchase = more churn
        )
        y = (np.random.random(n_customers) < churn_probability).astype(int)
        
        # ✅ CORRETTO: Features engineering senza target leakage
        X = pd.DataFrame({
            # Demographics (sempre disponibili)
            'age': ages,
            'income': incomes,
            'region_north': (regions == 'North').astype(int),
            'region_south': (regions == 'South').astype(int),
            'region_east': (regions == 'East').astype(int),
            'region_west': (regions == 'West').astype(int),
            
            # Historical behavior (informazioni passate)
            'past_purchases': past_purchases,
            'avg_order_value': avg_order_value,
            'days_since_last_purchase': days_since_last_purchase,
            
            # Derived features (basate su info passate)
            'customer_value': past_purchases * avg_order_value,  # Lifetime value
            'purchase_frequency': past_purchases / (days_since_last_purchase + 1),
            'high_value_customer': (avg_order_value > np.median(avg_order_value)).astype(int),
            'recent_customer': (days_since_last_purchase < 7).astype(int),
            
            # Interaction features (OK se basate su info passate)
            'age_income_interaction': ages * np.log(incomes),
            'value_frequency_ratio': avg_order_value / (past_purchases + 1),
        })
        
        # ✅ CORRETTO: Verifica che non ci siano correlazioni sospette con target
        target_correlations = X.corrwith(pd.Series(y)).abs()
        max_correlation = target_correlations.max()
        
        # Le correlazioni dovrebbero essere ragionevoli (< 0.8)
        assert max_correlation < 0.8, f"Suspicious correlation detected: {max_correlation}"
        
        # ✅ CORRETTO: Split normale (non temporale per questo caso)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # ✅ CORRETTO: Preprocessing corretto
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ✅ CORRETTO: Training
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Valutazione
        train_score = accuracy_score(y_train, model.predict(X_train_scaled))
        test_score = accuracy_score(y_test, model.predict(X_test_scaled))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'max_target_correlation': max_correlation,
            'feature_count': len(X.columns)
        }
    
    def clean_duplicate_detection_example(self):
        """
        ✅ CORRETTO: Gestione duplicati senza contaminazione
        Dimostra rimozione corretta di duplicati prima dello split
        """
        # Genera dataset con alcuni duplicati intenzionali
        n_samples = 1500
        n_features = 8
        
        # Features base
        X_base = np.random.randn(n_samples, n_features)
        y_base = (X_base[:, 0] + X_base[:, 1] > 0).astype(int)
        
        # ✅ CORRETTO: Aggiungi duplicati prima di split
        n_duplicates = 150
        duplicate_indices = np.random.choice(n_samples, n_duplicates, replace=True)
        
        X_with_duplicates = np.vstack([X_base, X_base[duplicate_indices]])
        y_with_duplicates = np.concatenate([y_base, y_base[duplicate_indices]])
        
        # Converti in DataFrame per gestione duplicati
        X_df = pd.DataFrame(X_with_duplicates, columns=[f'feature_{i}' for i in range(n_features)])
        y_series = pd.Series(y_with_duplicates)
        
        # ✅ CORRETTO: Rimuovi duplicati PRIMA dello split
        print(f"Samples before duplicate removal: {len(X_df)}")
        
        # Identifica duplicati
        duplicated_mask = X_df.duplicated(keep='first')
        
        # Rimuovi duplicati
        X_clean = X_df[~duplicated_mask].reset_index(drop=True)
        y_clean = y_series[~duplicated_mask].reset_index(drop=True)
        
        print(f"Samples after duplicate removal: {len(X_clean)}")
        print(f"Duplicates removed: {duplicated_mask.sum()}")
        
        # ✅ CORRETTO: Split dopo rimozione duplicati
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # ✅ CORRETTO: Verifica che non ci siano duplicati tra train e test
        train_tuples = set(X_train.apply(tuple, axis=1))
        test_tuples = set(X_test.apply(tuple, axis=1))
        contamination = train_tuples.intersection(test_tuples)
        
        assert len(contamination) == 0, f"Found {len(contamination)} duplicates between train and test"
        
        # ✅ CORRETTO: Training normale
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Valutazione
        train_score = accuracy_score(y_train, model.predict(X_train))
        test_score = accuracy_score(y_test, model.predict(X_test))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'duplicates_removed': duplicated_mask.sum(),
            'final_samples': len(X_clean),
            'train_test_contamination': len(contamination)
        }
    
    def clean_cross_validation_example(self):
        """
        ✅ CORRETTO: Cross-validation senza contaminazione
        Dimostra CV corretto con preprocessing all'interno dei fold
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.pipeline import Pipeline
        
        # Dataset semplice
        n_samples = 800
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.normal(0, 0.5, n_samples) > 0).astype(int)
        
        # ✅ CORRETTO: Pipeline che fa preprocessing all'interno di ogni fold
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Preprocessing
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        
        # ✅ CORRETTO: Cross-validation stratificata
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # ✅ CORRETTO: CV scores senza data leakage
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        # ✅ CORRETTO: Final model su tutto il dataset per production
        pipeline.fit(X, y)
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'pipeline_used': True
        }
    
    def clean_feature_selection_example(self):
        """
        ✅ CORRETTO: Feature selection senza target leakage
        Dimostra selezione features corretta all'interno di CV
        """
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.pipeline import Pipeline
        
        # Dataset con features rumorose
        n_samples = 600
        n_features = 50
        n_informative = 10
        
        # Features informative
        X_informative = np.random.randn(n_samples, n_informative)
        y = (X_informative[:, 0] + X_informative[:, 1] > 0).astype(int)
        
        # Features rumorose
        X_noise = np.random.randn(n_samples, n_features - n_informative)
        
        # Combina features
        X = np.hstack([X_informative, X_noise])
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        
        # ✅ CORRETTO: Pipeline con feature selection all'interno di CV
        pipeline = Pipeline([
            ('feature_selector', SelectKBest(f_classif, k=15)),  # Seleziona top 15 features
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        
        # ✅ CORRETTO: CV che include feature selection in ogni fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        # ✅ CORRETTO: Fit finale per vedere quali features sono selezionate
        pipeline.fit(X, y)
        selected_features = pipeline.named_steps['feature_selector'].get_support()
        selected_feature_names = X.columns[selected_features].tolist()
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'selected_features': len(selected_feature_names),
            'feature_selection_in_cv': True,
            'informative_features_selected': sum(1 for f in selected_feature_names if int(f.split('_')[1]) < n_informative)
        }


def run_all_clean_examples():
    """Esegue tutti gli esempi di codice pulito per test set contamination"""
    
    print("🧪 CLEAN CODE EXAMPLES - Test Set Contamination Prevention")
    print("=" * 70)
    
    cleaner = CleanContaminationPrevention()
    
    # Test 1: Temporal split
    print("\n1. ✅ Clean Temporal Split for Time Series")
    result1 = cleaner.clean_temporal_split_example()
    print(f"   Train score: {result1['train_score']:.3f}")
    print(f"   Test score: {result1['test_score']:.3f}")
    print(f"   Temporal split used: {result1['temporal_split']}")
    print(f"   No future leakage: {result1['no_future_leakage']}")
    
    # Test 2: Feature engineering
    print("\n2. ✅ Clean Feature Engineering without Target Leakage")
    result2 = cleaner.clean_feature_engineering_example()
    print(f"   Train score: {result2['train_score']:.3f}")
    print(f"   Test score: {result2['test_score']:.3f}")
    print(f"   Max target correlation: {result2['max_target_correlation']:.3f}")
    print(f"   Features created: {result2['feature_count']}")
    
    # Test 3: Duplicate handling
    print("\n3. ✅ Clean Duplicate Removal before Split")
    result3 = cleaner.clean_duplicate_detection_example()
    print(f"   Train score: {result3['train_score']:.3f}")
    print(f"   Test score: {result3['test_score']:.3f}")
    print(f"   Duplicates removed: {result3['duplicates_removed']}")
    print(f"   Train/test contamination: {result3['train_test_contamination']}")
    
    # Test 4: Cross-validation
    print("\n4. ✅ Clean Cross-Validation with Pipeline")
    result4 = cleaner.clean_cross_validation_example()
    print(f"   CV mean score: {result4['cv_mean']:.3f} ± {result4['cv_std']:.3f}")
    print(f"   Pipeline used: {result4['pipeline_used']}")
    
    # Test 5: Feature selection
    print("\n5. ✅ Clean Feature Selection within CV")
    result5 = cleaner.clean_feature_selection_example()
    print(f"   CV mean score: {result5['cv_mean']:.3f} ± {result5['cv_std']:.3f}")
    print(f"   Features selected: {result5['selected_features']}")
    print(f"   Informative features found: {result5['informative_features_selected']}")
    
    print("\n" + "=" * 70)
    print("✅ All clean examples completed successfully!")
    print("🎯 These patterns should NOT trigger contamination warnings")
    print("=" * 70)


if __name__ == "__main__":
    """Esegui esempi di codice pulito"""
    run_all_clean_examples()