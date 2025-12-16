# -*- coding: utf-8 -*-
"""
Prétraitement Adaptatif - SMOTE appliqué uniquement si déséquilibre détecté
Division temporelle respectée (batches par date)
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def load_params():
    """Charge les paramètres"""
    with open('params.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f).get('preprocess', {})


def check_class_balance(y, batch_name, threshold=1.5):
    """
    Vérifie si les classes sont déséquilibrées
    
    Args:
        y: Target variable
        batch_name: Nom du batch pour logging
        threshold: Seuil de déséquilibre (ratio max/min)
    
    Returns:
        tuple: (is_imbalanced, ratio, counts)
    """
    class_counts = y.value_counts()
    ratio = class_counts.max() / class_counts.min()
    
    print(f"\n[*] ANALYSE DÉSÉQUILIBRE - {batch_name}")
    print(f"{'='*60}")
    for cls, count in class_counts.items():
        pct = 100 * count / len(y)
        print(f"   Classe '{cls}': {count:,} ({pct:.1f}%)")
    print(f"   Ratio déséquilibre: {ratio:.2f}:1")
    
    is_imbalanced = ratio >= threshold
    
    if is_imbalanced:
        print(f"   ⚠️  DÉSÉQUILIBRE DÉTECTÉ (ratio >= {threshold})")
        print(f"   → SMOTE sera appliqué")
    else:
        print(f"   ✅ Classes équilibrées (ratio < {threshold})")
        print(f"   → SMOTE non nécessaire")
    
    return is_imbalanced, ratio, class_counts.to_dict()


def apply_smote_if_needed(X, y, batch_name, params):
    """
    Applique SMOTE uniquement si déséquilibre détecté
    
    Args:
        X: Features
        y: Target (encoded)
        batch_name: Nom du batch
        params: Paramètres yaml
    
    Returns:
        tuple: (X_resampled, y_resampled, smote_applied)
    """
    imbalance_threshold = params.get('imbalance_threshold', 1.5)
    is_imbalanced, ratio, counts = check_class_balance(
        pd.Series(y), 
        batch_name, 
        imbalance_threshold
    )
    
    if not is_imbalanced:
        print(f"   → Données inchangées")
        return X, y, False
    
    # Appliquer SMOTE
    print(f"\n[*] APPLICATION DE SMOTE")
    print(f"   Avant: {len(y)} samples")
    
    smote_strategy = params.get('smote_strategy', 'auto')
    
    if smote_strategy == 'auto':
        target_ratio = 0.8
    elif smote_strategy == 'balanced':
        target_ratio = 1.0
    else:
        target_ratio = float(smote_strategy)
    
    try:
        smote = SMOTE(
            sampling_strategy=target_ratio,
            random_state=params.get('random_state', 42),
            k_neighbors=min(5, min(counts.values()) - 1)
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"   Après: {len(y_resampled)} samples")
        
        new_counts = pd.Series(y_resampled).value_counts()
        for cls, count in new_counts.items():
            pct = 100 * count / len(y_resampled)
            print(f"   Classe {cls}: {count:,} ({pct:.1f}%)")
        
        new_ratio = new_counts.max() / new_counts.min()
        print(f"   Nouveau ratio: {new_ratio:.2f}:1")
        print(f"   ✅ SMOTE appliqué avec succès")
        
        return X_resampled, y_resampled, True
        
    except Exception as e:
        print(f"   ⚠️  Erreur SMOTE: {e}")
        print(f"   → Utilisation des données originales")
        return X, y, False


def preprocess_all_batches_global(train_dir, output_dir, params):
    """
    Prétraitement GLOBAL pour cohérence des transformers
    puis application ADAPTATIVE de SMOTE par batch
    """
    print(f"\n{'='*80}")
    print("[*] ÉTAPE 1: CHARGEMENT ET ANALYSE DE TOUS LES BATCHES")
    print(f"{'='*80}")
    
    # Charger tous les batches
    batch_files = sorted(Path(train_dir).glob('batch_*.csv'))
    if not batch_files:
        raise FileNotFoundError(f"Aucun batch trouvé dans {train_dir}")
    
    all_dfs = []
    batch_names = []
    batch_info = {}
    
    for batch_file in batch_files:
        df = pd.read_csv(batch_file)
        batch_name = batch_file.stem
        
        if 'satisfaction' in df.columns:
            target_dist = df['satisfaction'].value_counts()
            ratio = target_dist.max() / target_dist.min() if len(target_dist) > 1 else 1.0
            
            print(f"\n[+] {batch_name}: {len(df):,} lignes")
            for cls, count in target_dist.items():
                pct = 100 * count / len(df)
                print(f"    {cls}: {count:,} ({pct:.1f}%)")
            print(f"    Ratio: {ratio:.2f}:1")
            
            batch_info[batch_name] = {
                'size': len(df),
                'distribution': target_dist.to_dict(),
                'ratio': ratio
            }
        
        df['original_batch'] = batch_name
        all_dfs.append(df)
        batch_names.append(batch_name)
    
    df_full = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[+] TOTAL: {len(df_full):,} lignes x {df_full.shape[1]} colonnes")
    
    # Preprocessing global
    print(f"\n{'='*80}")
    print("[*] ÉTAPE 2: CRÉATION DES TRANSFORMERS GLOBAUX")
    print(f"{'='*80}")
    
    df_full = df_full.drop(columns=['id'], errors='ignore')
    target_col = 'satisfaction'
    
    if target_col not in df_full.columns:
        raise ValueError(f"Colonne '{target_col}' non trouvée")
    
    y = df_full[target_col]
    X = df_full.drop(columns=[target_col, 'original_batch'])
    batch_ids = df_full['original_batch']
    
    # Valeurs manquantes
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == 'object':
                mode_val = X[col].mode()
                if len(mode_val) > 0:
                    X[col] = X[col].fillna(mode_val[0])
            else:
                X[col] = X[col].fillna(X[col].median())
    
    # Encoder catégorielles
    encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if X[col].nunique() == 2:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        else:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    
    print(f"[+] Features après encoding: {X.shape[1]}")
    
    # Encoder target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print(f"[+] Classes: {target_encoder.classes_} → {np.unique(y_encoded)}")
    
    # Outliers
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
    
    # Normaliser
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"[+] Normalisation appliquée")
    
    # Feature selection
    variance_threshold = params.get('variance_threshold', 0.005)
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    X_var = variance_selector.fit_transform(X_scaled_df)
    selected_after_var = X_scaled_df.columns[variance_selector.get_support()]
    
    print(f"[+] Features après variance threshold: {len(selected_after_var)}")
    
    k_best = params.get('k_best_features', 30)
    k = min(k_best, len(selected_after_var))
    kbest_selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = kbest_selector.fit_transform(X_var, y_encoded)
    selected_features = selected_after_var[kbest_selector.get_support()]
    
    print(f"[+] Features finales (K-best): {len(selected_features)}")
    
    X_final = pd.DataFrame(X_selected, columns=selected_features)
    X_final['original_batch'] = batch_ids.values
    
    # Sauvegarder les transformers globaux
    os.makedirs(output_dir, exist_ok=True)
    
    global_preprocessing = {
        'scaler': scaler,
        'variance_selector': variance_selector,
        'kbest_selector': kbest_selector,
        'target_encoder': target_encoder,
        'encoders': encoders,
        'selected_features': selected_features.tolist(),
        'feature_names': X.columns.tolist(),
        'batch_info': batch_info
    }
    
    joblib.dump(global_preprocessing, f'{output_dir}/global_preprocessing.pkl')
    print(f"\n[+] Sauvegarde: global_preprocessing.pkl")
    
    # Traiter chaque batch avec SMOTE adaptatif
    print(f"\n{'='*80}")
    print("[*] ÉTAPE 3: TRAITEMENT ADAPTATIF PAR BATCH")
    print(f"{'='*80}")
    
    batch_metadata = {}
    
    for batch_name in batch_names:
        print(f"\n{'─'*80}")
        print(f"[*] BATCH: {batch_name}")
        print(f"{'─'*80}")
        
        mask = X_final['original_batch'] == batch_name
        X_batch = X_final[mask].drop(columns=['original_batch']).values
        y_batch = y_encoded[mask]
        
        print(f"[+] Données originales: {len(y_batch):,} samples")
        
        # Appliquer SMOTE si nécessaire
        X_batch_final, y_batch_final, smote_applied = apply_smote_if_needed(
            X_batch, y_batch, batch_name, params
        )
        
        # Sauvegarder
        X_batch_df = pd.DataFrame(X_batch_final, columns=selected_features)
        y_batch_series = pd.Series(y_batch_final, name=target_col)
        
        X_batch_df.to_csv(f'{output_dir}/{batch_name}_X_processed.csv', index=False)
        y_batch_series.to_csv(f'{output_dir}/{batch_name}_y_processed.csv', index=False)
        
        joblib.dump(global_preprocessing, f'{output_dir}/{batch_name}_preprocessing.pkl')
        
        batch_metadata[batch_name] = {
            'original_size': int(mask.sum()),
            'final_size': len(y_batch_final),
            'smote_applied': smote_applied,
            'original_distribution': batch_info[batch_name]['distribution'],
            'original_ratio': batch_info[batch_name]['ratio'],
            'final_distribution': pd.Series(y_batch_final).value_counts().to_dict()
        }
        
        print(f"[+] Sauvegarde: {batch_name}_X_processed.csv")
        print(f"[+] Sauvegarde: {batch_name}_y_processed.csv")
    
    # Sauvegarder le résumé
    import json
    
    summary = {
        'total_samples': len(df_full),
        'n_features': len(selected_features),
        'n_batches': len(batch_names),
        'batches': batch_metadata,
        'preprocessing_params': {
            'k_best_features': k_best,
            'variance_threshold': variance_threshold,
            'imbalance_threshold': params.get('imbalance_threshold', 1.5),
            'smote_strategy': params.get('smote_strategy', 'auto')
        }
    }
    
    with open(f'{output_dir}/preprocessing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Résumé
    print(f"\n{'='*80}")
    print("[*] RÉSUMÉ DU PREPROCESSING")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame(batch_metadata).T
    summary_df = summary_df[['original_size', 'final_size', 'smote_applied', 'original_ratio']]
    summary_df.columns = ['Taille Orig.', 'Taille Finale', 'SMOTE', 'Ratio Orig.']
    
    print("\n" + summary_df.to_string())
    
    print(f"\n[+] Total features: {len(selected_features)}")
    print(f"[+] Batches avec SMOTE: {sum(summary_df['SMOTE'])}/{len(batch_names)}")
    
    print(f"\n{'='*80}")
    print("[+] PREPROCESSING TERMINÉ")
    print(f"{'='*80}")
    
    return X_final, y_encoded, global_preprocessing, batch_metadata


def main():
    """Pipeline principal"""
    print("="*80)
    print("PREPROCESSING ADAPTATIF PAR BATCH")
    print("SMOTE appliqué uniquement si déséquilibre détecté")
    print("="*80)
    
    params = load_params()
    train_dir = params.get('train_dir', 'data/raw/train')
    output_dir = params.get('output_dir', 'data/processed')
    
    X_final, y_encoded, preprocessing, metadata = preprocess_all_batches_global(
        train_dir, output_dir, params
    )
    
    print("\n[✓] Preprocessing adaptatif terminé avec succès!")


if __name__ == "__main__":
    main()