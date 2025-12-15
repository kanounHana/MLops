# -*- coding: utf-8 -*-
"""
Prétraitement des données - Satisfaction Passagers Aériens
Adapté pour pipeline DVC avec gestion des batches
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
    """Charge les paramètres depuis params.yaml"""
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        return params.get('preprocess', {})
    except FileNotFoundError:
        print("⚠️  params.yaml non trouvé, utilisation des paramètres par défaut")
        return {}


def load_batches(train_dir='data/raw/train', test_dir='data/raw/test'):
    """Charge tous les batches de données"""
    print("="*80)
    print("✈️ PRÉTRAITEMENT - SATISFACTION PASSAGERS AÉRIENS")
    print("="*80)
    
    print("\n" + "="*80)
    print("📁 SECTION 1: CHARGEMENT DES DONNÉES")
    print("="*80)
    
    train_batches = []
    
    # Charger tous les fichiers batch_*.csv du dossier train
    batch_files = sorted(Path(train_dir).glob('batch_*.csv'))
    
    if not batch_files:
        raise FileNotFoundError(f"Aucun fichier batch trouvé dans {train_dir}")
    
    for batch_file in batch_files:
        df = pd.read_csv(batch_file)
        train_batches.append(df)
        print(f"✅ {batch_file.name}: {len(df):,} lignes")
    
    # Concaténer tous les batches
    train_df = pd.concat(train_batches, ignore_index=True)
    
    # Charger les données de test
    test_file = Path(test_dir) / 'batch_test.csv'
    test_df = pd.read_csv(test_file)
    
    print(f"\n📊 Total Train: {train_df.shape[0]:,} lignes × {train_df.shape[1]} colonnes")
    print(f"📊 Test: {test_df.shape[0]:,} lignes × {test_df.shape[1]} colonnes")
    
    return train_df, test_df


def drop_useless_columns(train_df, test_df):
    """Supprime les colonnes inutiles"""
    print("\n" + "="*80)
    print("🎯 SECTION 2: SUPPRESSION DES COLONNES INUTILES")
    print("="*80)
    
    cols_to_drop = ['id']
    
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"✅ Colonnes supprimées: {cols_to_drop}")
    print(f"📊 Dimensions après suppression:")
    print(f"   Train: {train_df.shape[0]:,} lignes × {train_df.shape[1]} colonnes")
    print(f"   Test: {test_df.shape[0]:,} lignes × {test_df.shape[1]} colonnes")
    
    return train_df, test_df


def handle_missing_values(train_df, test_df):
    """Gère les valeurs manquantes"""
    print("\n" + "="*80)
    print("🔧 SECTION 3: GESTION DES VALEURS MANQUANTES")
    print("="*80)
    
    missing_train = train_df.isnull().sum()
    missing_test = test_df.isnull().sum()
    
    missing_cols_train = missing_train[missing_train > 0].index.tolist()
    missing_cols_test = missing_test[missing_test > 0].index.tolist()
    
    if missing_cols_train or missing_cols_test:
        print("⚠️  Colonnes avec valeurs manquantes:")
        
        for df_name, df_missing, missing_cols, df in [
            ("Train", missing_train, missing_cols_train, train_df),
            ("Test", missing_test, missing_cols_test, test_df)
        ]:
            if missing_cols:
                print(f"\n🔸 {df_name}:")
                for col in missing_cols:
                    missing_pct = (df_missing[col] / len(df)) * 100
                    print(f"   - {col}: {df_missing[col]} valeurs ({missing_pct:.2f}%)")
        
        print("\n🎯 Stratégie d'imputation:")
        
        # 1. Colonnes catégorielles: Mode
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        cat_missing = [col for col in categorical_cols if col in missing_cols_train or col in missing_cols_test]
        
        if cat_missing:
            print("   🔸 Catégorielles: Imputation par mode")
            for col in cat_missing:
                if col in train_df.columns:
                    mode_val = train_df[col].mode()[0]
                    train_df[col] = train_df[col].fillna(mode_val)
                if col in test_df.columns:
                    mode_val = test_df[col].mode()[0] if not test_df[col].mode().empty else train_df[col].mode()[0]
                    test_df[col] = test_df[col].fillna(mode_val)
        
        # 2. Colonnes numériques: Médiane
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        num_missing = [col for col in numeric_cols if col in missing_cols_train or col in missing_cols_test]
        
        if num_missing:
            print("   🔸 Numériques: Imputation par médiane")
            for col in num_missing:
                if col in train_df.columns:
                    median_val = train_df[col].median()
                    train_df[col] = train_df[col].fillna(median_val)
                if col in test_df.columns:
                    median_val = test_df[col].median() if not test_df[col].isnull().all() else train_df[col].median()
                    test_df[col] = test_df[col].fillna(median_val)
        
        print("\n✅ Après imputation:")
        print(f"   Train - Valeurs manquantes: {train_df.isnull().sum().sum()}")
        print(f"   Test - Valeurs manquantes: {test_df.isnull().sum().sum()}")
    else:
        print("✅ Aucune valeur manquante détectée!")
    
    return train_df, test_df


def encode_categorical_features(train_df, test_df):
    """Encode les variables catégorielles"""
    print("\n" + "="*80)
    print("🏷️ SECTION 4: ENCODAGE DES VARIABLES CATÉGORIELLES")
    print("="*80)
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    target_col = 'satisfaction'
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    print(f"🔍 Variables catégorielles à encoder: {categorical_cols}")
    
    encoders = {}
    
    for col in categorical_cols:
        print(f"\n🔸 Encodage de '{col}':")
        
        if train_df[col].nunique() == 2:
            print(f"   ⚡ Encodage binaire (Label Encoding)")
            le = LabelEncoder()
            
            train_df[col] = le.fit_transform(train_df[col])
            
            test_categories = set(test_df[col].unique())
            train_categories = set(le.classes_)
            
            if not test_categories.issubset(train_categories):
                print(f"   ⚠️  Catégories inconnues dans test: {test_categories - train_categories}")
                most_frequent = train_df[col].mode()[0]
                test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[most_frequent])
            
            test_df[col] = le.transform(test_df[col])
            encoders[col] = le
            
            print(f"   📊 Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        else:
            print(f"   🎯 One-Hot Encoding ({train_df[col].nunique()} catégories)")
            
            dummies_train = pd.get_dummies(train_df[col], prefix=col, drop_first=True)
            dummies_test = pd.get_dummies(test_df[col], prefix=col, drop_first=True)
            
            missing_cols = set(dummies_train.columns) - set(dummies_test.columns)
            for c in missing_cols:
                dummies_test[c] = 0
            
            dummies_test = dummies_test[dummies_train.columns]
            
            train_df = pd.concat([train_df.drop(columns=[col]), dummies_train], axis=1)
            test_df = pd.concat([test_df.drop(columns=[col]), dummies_test], axis=1)
            
            print(f"   📊 Colonnes créées: {list(dummies_train.columns)}")
    
    print(f"\n✅ Encodage terminé!")
    print(f"📊 Nouvelles dimensions:")
    print(f"   Train: {train_df.shape[0]:,} lignes × {train_df.shape[1]} colonnes")
    print(f"   Test: {test_df.shape[0]:,} lignes × {test_df.shape[1]} colonnes")
    
    return train_df, test_df, encoders


def encode_target(train_df, test_df):
    """Encode la variable cible"""
    print("\n" + "="*80)
    print("🎯 SECTION 5: ENCODAGE DE LA VARIABLE CIBLE")
    print("="*80)
    
    target_col = 'satisfaction'
    
    print("🔸 Encodage de la variable cible 'satisfaction'")
    target_encoder = LabelEncoder()
    train_df[target_col] = target_encoder.fit_transform(train_df[target_col])
    
    if target_col in test_df.columns:
        test_df[target_col] = target_encoder.transform(test_df[target_col])
    
    print(f"📊 Mapping cible: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
    print(f"   - 0: {target_encoder.classes_[0]}")
    print(f"   - 1: {target_encoder.classes_[1]}")
    
    return train_df, test_df, target_encoder


def handle_outliers(train_df, test_df):
    """Gère les outliers"""
    print("\n" + "="*80)
    print("📈 SECTION 6: GESTION DES OUTLIERS")
    print("="*80)
    
    target_col = 'satisfaction'
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    skewed_cols = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
    
    print("🔍 Gestion des outliers via winsorization ou transformation:")
    
    for col in numeric_cols:
        if col in skewed_cols:
            print(f"🔸 {col}: Forte asymétrie détectée → Transformation log")
            
            train_df[f'{col}_log'] = np.log1p(train_df[col])
            test_df[f'{col}_log'] = np.log1p(test_df[col])
            
            train_df = train_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])
        else:
            Q1 = train_df[col].quantile(0.01)
            Q3 = train_df[col].quantile(0.99)
            
            outliers_count = ((train_df[col] < Q1) | (train_df[col] > Q3)).sum()
            if outliers_count > 0 and outliers_count < len(train_df) * 0.05:
                print(f"🔸 {col}: {outliers_count} outliers → Winsorization (1%-99%)")
                
                train_df[col] = train_df[col].clip(Q1, Q3)
                test_df[col] = test_df[col].clip(Q1, Q3)
    
    print("\n✅ Gestion des outliers terminée!")
    
    return train_df, test_df


def scale_features(train_df, test_df):
    """Normalise les features"""
    print("\n" + "="*80)
    print("⚖️ SECTION 7: NORMALISATION/STANDARDISATION")
    print("="*80)
    
    target_col = 'satisfaction'
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col], errors='ignore')
    y_test = test_df[target_col] if target_col in test_df.columns else None
    
    print(f"🔍 {len(X_train.columns)} features à normaliser")
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("✅ Normalisation avec RobustScaler terminée!")
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler


def select_features(X_train, X_test, y_train, k=20):
    """Sélectionne les meilleures features"""
    print("\n" + "="*80)
    print("🔍 SECTION 8: SÉLECTION DE FEATURES")
    print("="*80)
    
    # 1. Variance threshold
    print("1️⃣  Suppression des features à variance nulle:")
    selector_variance = VarianceThreshold(threshold=0.01)
    X_train_var = selector_variance.fit_transform(X_train)
    X_test_var = selector_variance.transform(X_test)
    
    selected_features = X_train.columns[selector_variance.get_support()]
    print(f"   ✅ {len(selected_features)} features conservées sur {X_train.shape[1]}")
    
    # 2. SelectKBest
    print("\n2️⃣  Sélection basée sur ANOVA F-value:")
    k = min(k, len(selected_features))
    selector_kbest = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector_kbest.fit_transform(X_train[selected_features], y_train)
    X_test_selected = selector_kbest.transform(X_test[selected_features])
    
    selected_features_kbest = selected_features[selector_kbest.get_support()]
    
    print(f"   ✅ Top {k} features sélectionnées:")
    scores = selector_kbest.scores_
    indices = np.argsort(scores)[::-1]
    
    for i in range(min(10, len(scores))):
        idx = indices[i]
        print(f"      {i+1:2d}. {selected_features[idx]:30} : {scores[idx]:.2f}")
    
    X_train_final = pd.DataFrame(X_train_selected, columns=selected_features_kbest, index=X_train.index)
    X_test_final = pd.DataFrame(X_test_selected, columns=selected_features_kbest, index=X_test.index)
    
    print(f"\n📊 Dimensions finales:")
    print(f"   X_train: {X_train_final.shape}")
    print(f"   X_test: {X_test_final.shape}")
    
    return X_train_final, X_test_final, selector_variance, selector_kbest, selected_features_kbest


def save_processed_data(X_train, X_test, y_train, y_test, scaler, variance_selector, 
                       kbest_selector, target_encoder, encoders, selected_features,
                       output_dir='data/processed'):
    """Sauvegarde les données prétraitées"""
    print("\n" + "="*80)
    print("💾 SECTION 9: SAUVEGARDE DES DONNÉES PRÉTRAITÉES")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f'{output_dir}/X_train_processed.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test_processed.csv', index=False)
    
    y_train.to_csv(f'{output_dir}/y_train_processed.csv', index=False)
    if y_test is not None:
        y_test.to_csv(f'{output_dir}/y_test_processed.csv', index=False)
    
    preprocessing_objects = {
        'scaler': scaler,
        'variance_selector': variance_selector,
        'kbest_selector': kbest_selector,
        'target_encoder': target_encoder,
        'encoders': encoders,
        'selected_features': selected_features.tolist()
    }
    
    joblib.dump(preprocessing_objects, f'{output_dir}/preprocessing_objects.pkl')
    
    print("✅ Données sauvegardées avec succès!")
    print(f"📁 Dossier: {output_dir}")
    print(f"📄 Fichiers générés:")
    print(f"   - X_train_processed.csv")
    print(f"   - X_test_processed.csv")
    print(f"   - y_train_processed.csv")
    if y_test is not None:
        print(f"   - y_test_processed.csv")
    print(f"   - preprocessing_objects.pkl")


def print_summary(initial_shape, final_shape, X_train, X_test):
    """Affiche le résumé du prétraitement"""
    print("\n" + "="*80)
    print("📊 SECTION 10: RÉSUMÉ DU PRÉTRAITEMENT")
    print("="*80)
    
    print("🎯 ÉTAPES EFFECTUÉES:")
    print("""
1. ✅ Chargement des données (batches)
2. ✅ Suppression des colonnes inutiles (id)
3. ✅ Gestion des valeurs manquantes (imputation médiane/mode)
4. ✅ Encodage des variables catégorielles
5. ✅ Encodage de la variable cible (satisfaction → 0/1)
6. ✅ Gestion des outliers
7. ✅ Normalisation avec RobustScaler
8. ✅ Sélection de features
9. ✅ Sauvegarde des données prétraitées
""")
    
    print("📈 STATISTIQUES FINALES:")
    print(f"   - Nombre de features initial: {initial_shape[1] - 1}")
    print(f"   - Nombre de features final: {final_shape[1]}")
    reduction = ((initial_shape[1] - 1 - final_shape[1]) / (initial_shape[1] - 1) * 100)
    print(f"   - Réduction: {reduction:.1f}%")
    print(f"   - Taille échantillon train: {X_train.shape[0]:,}")
    print(f"   - Taille échantillon test: {X_test.shape[0]:,}")
    
    print("\n🔍 APERÇU DES DONNÉES PRÉTRAITÉES (X_train):")
    print(X_train.head())
    
    print("\n" + "="*80)
    print("✅ PRÉTRAITEMENT TERMINÉ - PRÊT POUR LA MODÉLISATION!")
    print("="*80)


def main():
    """Pipeline principal de preprocessing"""
    # Charger les paramètres
    params = load_params()
    
    train_dir = params.get('train_dir', 'data/raw/train')
    test_dir = params.get('test_dir', 'data/raw/test')
    output_dir = params.get('output_dir', 'data/processed')
    k_features = params.get('k_best_features', 20)
    
    # 1. Charger les données
    train_df, test_df = load_batches(train_dir, test_dir)
    initial_shape = train_df.shape
    
    # 2. Supprimer colonnes inutiles
    train_df, test_df = drop_useless_columns(train_df, test_df)
    
    # 3. Gérer valeurs manquantes
    train_df, test_df = handle_missing_values(train_df, test_df)
    
    # 4. Encoder variables catégorielles
    train_df, test_df, encoders = encode_categorical_features(train_df, test_df)
    
    # 5. Encoder variable cible
    train_df, test_df, target_encoder = encode_target(train_df, test_df)
    
    # 6. Gérer outliers
    train_df, test_df = handle_outliers(train_df, test_df)
    
    # 7. Normaliser
    X_train, X_test, y_train, y_test, scaler = scale_features(train_df, test_df)
    
    # 8. Sélectionner features
    X_train_final, X_test_final, variance_selector, kbest_selector, selected_features = \
        select_features(X_train, X_test, y_train, k=k_features)
    
    # 9. Sauvegarder
    save_processed_data(X_train_final, X_test_final, y_train, y_test, 
                       scaler, variance_selector, kbest_selector, 
                       target_encoder, encoders, selected_features, output_dir)
    
    # 10. Résumé
    print_summary(initial_shape, X_train_final.shape, X_train_final, X_test_final)


if __name__ == "__main__":
    main()