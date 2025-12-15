# -*- coding: utf-8 -*-
"""
Prétraitement par Batch - Chaque batch est traité indépendamment
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


def load_params():
    """Charge les paramètres"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f).get('preprocess', {})


def preprocess_single_batch(batch_path, output_dir, batch_name):
    """Prétraite UN SEUL batch"""
    print(f"\n{'='*80}")
    print(f"[*] PREPROCESSING: {batch_name}")
    print(f"{'='*80}")
    
    # 1. Charger
    df = pd.read_csv(batch_path)
    print(f"[+] Charge: {len(df):,} lignes x {df.shape[1]} colonnes")
    
    # 2. Supprimer id
    df = df.drop(columns=['id'], errors='ignore')
    
    # 3. Séparer target
    target_col = 'satisfaction'
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        raise ValueError(f"Colonne '{target_col}' non trouvée")
    
    # 4. Valeurs manquantes
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == 'object':
                mode_val = X[col].mode()
                if len(mode_val) > 0:
                    X[col] = X[col].fillna(mode_val[0])
            else:
                X[col] = X[col].fillna(X[col].median())
    
    # 5. Encoder catégorielles
    encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if X[col].nunique() == 2:
            # Label Encoding binaire
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        else:
            # One-Hot Encoding
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    
    # 6. Encoder target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # 7. Outliers
    skewed_cols = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
    for col in skewed_cols:
        if col in X.columns:
            X[f'{col}_log'] = np.log1p(X[col])
            X = X.drop(columns=[col])
    
    winsorize_cols = ['Age', 'Flight Distance', 'Gate location', 'Food and drink', 
                     'Seat comfort', 'Inflight entertainment', 'On-board service',
                     'Leg room service', 'Checkin service', 'Inflight service', 'Cleanliness']
    
    for col in winsorize_cols:
        if col in X.columns:
            Q1 = X[col].quantile(0.01)
            Q3 = X[col].quantile(0.99)
            X[col] = X[col].clip(Q1, Q3)
    
    # 8. Normaliser
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 9. Feature selection
    variance_selector = VarianceThreshold(threshold=0.01)
    X_var = variance_selector.fit_transform(X_scaled_df)
    selected_after_var = X_scaled_df.columns[variance_selector.get_support()]
    
    k = min(20, len(selected_after_var))
    kbest_selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = kbest_selector.fit_transform(X_var, y_encoded)
    selected_features = selected_after_var[kbest_selector.get_support()]
    
    X_final = pd.DataFrame(X_selected, columns=selected_features)
    y_final = pd.Series(y_encoded, name=target_col)
    
    print(f"[+] Preprocessing termine: {X_final.shape[1]} features")
    
    # 10. Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    
    X_final.to_csv(f'{output_dir}/{batch_name}_X_processed.csv', index=False)
    y_final.to_csv(f'{output_dir}/{batch_name}_y_processed.csv', index=False)
    
    # Objets de preprocessing
    preprocessing_objects = {
        'scaler': scaler,
        'variance_selector': variance_selector,
        'kbest_selector': kbest_selector,
        'target_encoder': target_encoder,
        'encoders': encoders,
        'selected_features': selected_features.tolist(),
        'feature_names': X.columns.tolist()
    }
    
    joblib.dump(preprocessing_objects, f'{output_dir}/{batch_name}_preprocessing.pkl')
    
    print(f"[+] Sauvegarde:")
    print(f"   - {batch_name}_X_processed.csv")
    print(f"   - {batch_name}_y_processed.csv")
    print(f"   - {batch_name}_preprocessing.pkl")
    
    return X_final, y_final, preprocessing_objects


def main():
    """Prétraite tous les batches"""
    print("="*80)
    print("PREPROCESSING PAR BATCH")
    print("="*80)
    
    params = load_params()
    train_dir = params.get('train_dir', 'data/raw/train')
    output_dir = params.get('output_dir', 'data/processed')
    
    # Trouver tous les batches
    batch_files = sorted(Path(train_dir).glob('batch_*.csv'))
    
    if not batch_files:
        raise FileNotFoundError(f"Aucun batch trouve dans {train_dir}")
    
    print(f"\n[+] {len(batch_files)} batches detectes")
    
    # Prétraiter chaque batch
    for batch_file in batch_files:
        batch_name = batch_file.stem  # batch_1, batch_2, etc.
        preprocess_single_batch(batch_file, output_dir, batch_name)
    
    print("\n" + "="*80)
    print("[+] PREPROCESSING TERMINE POUR TOUS LES BATCHES")
    print("="*80)


if __name__ == "__main__":
    main()