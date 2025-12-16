# -*- coding: utf-8 -*-
"""
Preprocessing du TEST SET - Utilise les transformers du training
À exécuter APRÈS le training pour prétraiter batch_test.csv
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def preprocess_test_set(test_path, preprocessing_path, output_path):
    """
    Prétraite le test set avec les transformers du training
    
    IMPORTANT: 
    - Utilise les MÊMES transformers que le training
    - PAS de SMOTE sur le test
    - PAS de refit des encoders/scalers
    """
    print("="*80)
    print("PREPROCESSING DU TEST SET")
    print("="*80)
    
    # 1. Charger le test set
    print(f"\n[1] CHARGEMENT DES DONNÉES")
    print(f"    Fichier: {test_path}")
    
    df = pd.read_csv(test_path)
    print(f"    ✅ {len(df):,} lignes x {df.shape[1]} colonnes")
    
    # Sauvegarder les IDs
    if 'id' in df.columns:
        test_ids = df['id'].copy()
        df = df.drop(columns=['id'])
    else:
        test_ids = pd.Series(range(len(df)), name='id')
    
    print(f"    ✅ IDs sauvegardés: {len(test_ids):,}")
    
    # 2. Charger les transformers du training
    print(f"\n[2] CHARGEMENT DES TRANSFORMERS")
    print(f"    Fichier: {preprocessing_path}")
    
    if not Path(preprocessing_path).exists():
        raise FileNotFoundError(f"❌ {preprocessing_path} non trouvé!\n"
                              f"   Exécutez d'abord: dvc repro preprocess")
    
    preprocessing = joblib.load(preprocessing_path)
    
    scaler = preprocessing['scaler']
    variance_selector = preprocessing['variance_selector']
    kbest_selector = preprocessing['kbest_selector']
    encoders = preprocessing['encoders']
    selected_features = preprocessing['selected_features']
    feature_names_before_transform = preprocessing['feature_names']
    
    print(f"    ✅ Transformers chargés")
    print(f"    ✅ {len(selected_features)} features attendues")
    
    # 3. Vérifier la target (si présente, c'est un validation set)
    target_col = 'satisfaction'
    has_target = target_col in df.columns
    
    if has_target:
        print(f"\n[3] TARGET DÉTECTÉE (Validation Set)")
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        dist = y.value_counts()
        print(f"    Distribution:")
        for cls, count in dist.items():
            pct = 100 * count / len(y)
            print(f"      {cls}: {count:,} ({pct:.1f}%)")
    else:
        print(f"\n[3] PAS DE TARGET (Vrai Test Set)")
        X = df.copy()
        y = None
    
    print(f"    ✅ Features: {X.shape[1]}")
    
    # 4. Valeurs manquantes (MÊMES règles que training)
    print(f"\n[4] VALEURS MANQUANTES")
    
    missing_before = X.isnull().sum().sum()
    
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == 'object':
                mode_val = X[col].mode()
                if len(mode_val) > 0:
                    X[col] = X[col].fillna(mode_val[0])
                else:
                    X[col] = X[col].fillna('Unknown')
            else:
                X[col] = X[col].fillna(X[col].median())
    
    missing_after = X.isnull().sum().sum()
    print(f"    ✅ Valeurs manquantes: {missing_before} → {missing_after}")
    
    # 5. Encoder catégorielles (TRANSFORM avec encoders du training)
    print(f"\n[5] ENCODING DES VARIABLES CATÉGORIELLES")
    
    categorical_encoded = 0
    
    for col, encoder in encoders.items():
        if col in X.columns:
            try:
                X[col] = encoder.transform(X[col])
                categorical_encoded += 1
            except ValueError as e:
                # Catégories inconnues → utiliser la catégorie la plus fréquente du training
                print(f"    ⚠️  {col}: catégories inconnues détectées")
                known_classes = set(encoder.classes_)
                X[col] = X[col].apply(
                    lambda x: encoder.transform([x])[0] if x in known_classes 
                    else encoder.transform([encoder.classes_[0]])[0]
                )
                categorical_encoded += 1
    
    print(f"    ✅ {categorical_encoded} colonnes encodées avec LabelEncoder")
    
    # One-hot encoding pour colonnes non binaires
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"    🔄 One-hot encoding: {len(categorical_cols)} colonnes")
        
        for col in categorical_cols:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    
    print(f"    ✅ Total features après encoding: {X.shape[1]}")
    
    # 6. Aligner les colonnes avec le training
    print(f"\n[6] ALIGNEMENT DES COLONNES")
    
    # Ajouter colonnes manquantes
    for col in feature_names_before_transform:
        if col not in X.columns:
            X[col] = 0
    
    # Garder seulement les colonnes du training, dans le même ordre
    X = X[feature_names_before_transform]
    
    print(f"    ✅ {X.shape[1]} colonnes alignées")
    
    # 7. Traiter les outliers (MÊMES transformations que training)
    print(f"\n[7] TRAITEMENT DES OUTLIERS")
    
    # Log transform pour colonnes skewed
    skewed_cols = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
    for col in skewed_cols:
        if col in X.columns:
            X[f'{col}_log'] = np.log1p(X[col])
            X = X.drop(columns=[col])
            print(f"    ✅ Log transform: {col}")
    
    # Winsorization
    winsorize_cols = ['Age', 'Flight Distance', 'Gate location', 'Food and drink', 
                     'Seat comfort', 'Inflight entertainment', 'On-board service',
                     'Leg room service', 'Checkin service', 'Inflight service', 'Cleanliness']
    
    winsorized = 0
    for col in winsorize_cols:
        if col in X.columns:
            Q1 = X[col].quantile(0.01)
            Q3 = X[col].quantile(0.99)
            X[col] = X[col].clip(Q1, Q3)
            winsorized += 1
    
    print(f"    ✅ {winsorized} colonnes winsorisées")
    
    # 8. Normaliser (TRANSFORM avec scaler du training)
    print(f"\n[8] NORMALISATION")
    
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"    ✅ RobustScaler appliqué")
    
    # 9. Feature selection (TRANSFORM avec selectors du training)
    print(f"\n[9] SÉLECTION DES FEATURES")
    
    X_var = variance_selector.transform(X_scaled_df)
    print(f"    ✅ VarianceThreshold: {X_var.shape[1]} features")
    
    X_selected = kbest_selector.transform(X_var)
    print(f"    ✅ SelectKBest: {X_selected.shape[1]} features")
    
    # 10. DataFrame final
    X_final = pd.DataFrame(X_selected, columns=selected_features)
    
    print(f"\n[10] RÉSUMÉ")
    print("="*80)
    print(f"    Samples: {len(df):,} → {len(X_final):,}")
    print(f"    Features: {df.shape[1]} → {len(selected_features)}")
    print(f"    SMOTE appliqué: NON (test set)")
    print(f"    Transformers: Réutilisés du training ✅")
    
    # 11. Sauvegarder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    X_final.to_csv(output_path, index=False)
    print(f"\n✅ Sauvegardé: {output_path}")
    
    # Sauvegarder les IDs séparément
    ids_path = output_path.replace('_X_processed.csv', '_ids.csv')
    test_ids.to_csv(ids_path, index=False)
    print(f"✅ IDs sauvegardés: {ids_path}")
    
    # Sauvegarder y si présent
    if y is not None:
        y_path = output_path.replace('_X_processed.csv', '_y_processed.csv')
        
        # Encoder y avec le même encoder que le training
        target_encoder = preprocessing['target_encoder']
        y_encoded = target_encoder.transform(y)
        
        y_series = pd.Series(y_encoded, name=target_col)
        y_series.to_csv(y_path, index=False)
        print(f"✅ Target sauvegardée: {y_path}")
    
    return X_final, test_ids, y


def main():
    """Pipeline principal"""
    
    print("\n" + "🚀"*40)
    print("PREPROCESSING DU TEST SET")
    print("🚀"*40 + "\n")
    
    # Chemins
    test_path = 'data/raw/test/batch_test.csv'
    preprocessing_path = 'data/processed/global_preprocessing.pkl'
    output_path = 'data/processed/batch_test_X_processed.csv'
    
    # Vérifier que les fichiers existent
    if not Path(test_path).exists():
        print(f"❌ Fichier test non trouvé: {test_path}")
        print(f"\nFichiers disponibles dans data/raw/test/:")
        for f in Path('data/raw/test/').glob('*.csv'):
            print(f"   - {f.name}")
        return
    
    if not Path(preprocessing_path).exists():
        print(f"❌ Transformers non trouvés: {preprocessing_path}")
        print(f"\n💡 Exécutez d'abord:")
        print(f"   dvc repro preprocess")
        return
    
    # Prétraiter
    X_final, test_ids, y = preprocess_test_set(
        test_path, 
        preprocessing_path, 
        output_path
    )
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING DU TEST TERMINÉ")
    print("="*80)
    print(f"\nFichiers générés:")
    print(f"  - {output_path}")
    print(f"  - {output_path.replace('_X_processed.csv', '_ids.csv')}")
    if y is not None:
        print(f"  - {output_path.replace('_X_processed.csv', '_y_processed.csv')}")
    
    print(f"\n💡 Prochaine étape:")
    print(f"   Faire les prédictions avec ces données prétraitées")


if __name__ == "__main__":
    main()