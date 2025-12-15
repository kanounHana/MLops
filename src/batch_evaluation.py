# -*- coding: utf-8 -*-
"""
Évaluation par Batch - Analyse de performance temporelle
Entraîne sur chaque batch individuellement et évalue les performances
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def preprocess_single_batch(batch_df, preprocessing_objects):
    """Prétraite un batch unique avec les objets de preprocessing"""
    
    # Copier pour éviter modifications
    df = batch_df.copy()
    
    # 1. Supprimer id si existe
    df = df.drop(columns=['id'], errors='ignore')
    
    # 2. Gérer valeurs manquantes
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # 3. Encoder variables catégorielles (sauf target)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    target_col = 'satisfaction'
    
    # Sauvegarder la target avant de tout encoder
    if target_col in df.columns:
        target_encoder = preprocessing_objects['target_encoder']
        y = target_encoder.transform(df[target_col])
        df = df.drop(columns=[target_col])
    else:
        y = None
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    encoders = preprocessing_objects.get('encoders', {})
    
    for col in categorical_cols:
        if col in encoders:
            # Label encoding pour colonnes binaires
            le = encoders[col]
            unknown_mask = ~df[col].isin(le.classes_)
            if unknown_mask.any():
                df.loc[unknown_mask, col] = le.classes_[0]
            df[col] = le.transform(df[col])
        else:
            # One-hot encoding pour Class (3 catégories)
            if col == 'Class' and col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                # Assurer les bonnes colonnes
                if 'Class_Eco' not in dummies.columns:
                    dummies['Class_Eco'] = 0
                if 'Class_Eco Plus' not in dummies.columns:
                    dummies['Class_Eco Plus'] = 0
                df = pd.concat([df.drop(columns=[col]), dummies[['Class_Eco', 'Class_Eco Plus']]], axis=1)
    
    # 4. Gérer outliers - Winsorization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Appliquer winsorization pour les colonnes qui l'ont eu pendant l'entraînement
    winsorize_cols = ['Age', 'Flight Distance', 'Gate location', 'Food and drink', 
                     'Seat comfort', 'Inflight entertainment', 'On-board service',
                     'Leg room service', 'Checkin service', 'Inflight service', 'Cleanliness']
    
    for col in winsorize_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(Q1, Q3)
    
    # 5. Transformation log pour les délais
    skewed_cols = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
    for col in skewed_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
            df = df.drop(columns=[col])
    
    # 6. Normaliser AVANT la sélection de features
    # Le scaler a été fit sur toutes les colonnes, donc on doit d'abord normaliser
    scaler = preprocessing_objects['scaler']
    
    # Obtenir les noms de colonnes sur lesquelles le scaler a été fit
    # En vérifiant les feature_names_in_ du scaler
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = scaler.feature_names_in_
        
        # Ajouter les colonnes manquantes avec 0
        for feat in scaler_features:
            if feat not in df.columns:
                df[feat] = 0
        
        # Réorganiser dans le bon ordre
        df = df[scaler_features]
    
    # Appliquer le scaler
    X_scaled = scaler.transform(df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)
    
    # 7. Appliquer les sélecteurs de features
    variance_selector = preprocessing_objects['variance_selector']
    kbest_selector = preprocessing_objects['kbest_selector']
    
    # VarianceThreshold
    X_var = variance_selector.transform(X_scaled_df)
    selected_after_var = X_scaled_df.columns[variance_selector.get_support()]
    
    # SelectKBest
    X_kbest = kbest_selector.transform(X_var)
    
    # Obtenir les noms finaux des features
    expected_features = preprocessing_objects['selected_features']
    X = pd.DataFrame(X_kbest, columns=expected_features)
    
    return X, y


def train_on_batch(batch_name, X_train, y_train):
    """Entraîne un modèle XGBoost sur un batch"""
    
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_on_batch(model, X_test, y_test, batch_name):
    """Évalue le modèle sur un batch"""
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'batch': batch_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'n_samples': len(y_test),
        'n_positive': int((y_test == 1).sum()),
        'n_negative': int((y_test == 0).sum())
    }
    
    return metrics


def main():
    print("="*80)
    print("📊 ÉVALUATION PAR BATCH - ANALYSE TEMPORELLE")
    print("="*80)
    
    # Charger les objets de preprocessing
    preprocessing_objects = joblib.load('data/processed/preprocessing_objects.pkl')
    
    # Dossier des batches
    train_dir = Path('data/raw/train')
    batch_files = sorted(train_dir.glob('batch_*.csv'))
    
    print(f"\n📁 {len(batch_files)} batches détectés:")
    for bf in batch_files:
        print(f"   - {bf.name}")
    
    # ============================================================================
    # SCÉNARIO 1: Entraîner sur chaque batch, tester sur ce même batch
    # ============================================================================
    
    print("\n" + "="*80)
    print("📈 SCÉNARIO 1: Entraînement et test sur le MÊME batch")
    print("="*80)
    
    results_same_batch = []
    
    for batch_file in batch_files:
        batch_name = batch_file.stem  # batch_1, batch_2, etc.
        print(f"\n🔸 Traitement de {batch_name}...")
        
        # Charger et prétraiter le batch
        batch_df = pd.read_csv(batch_file)
        print(f"   📊 Taille: {len(batch_df):,} lignes")
        
        X, y = preprocess_single_batch(batch_df, preprocessing_objects)
        
        # Split train/test 80/20
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Entraîner
        model = train_on_batch(batch_name, X_train, y_train)
        
        # Évaluer
        metrics = evaluate_on_batch(model, X_test, y_test, batch_name)
        results_same_batch.append(metrics)
        
        print(f"   ✅ F1-Score: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Créer DataFrame des résultats
    df_same_batch = pd.DataFrame(results_same_batch)
    
    print("\n📊 RÉSULTATS - MÊME BATCH:")
    print("-"*80)
    print(df_same_batch[['batch', 'accuracy', 'f1', 'roc_auc', 'n_samples']].to_string(index=False))
    
    # ============================================================================
    # SCÉNARIO 2: Entraîner sur batch_1, tester sur tous les autres
    # ============================================================================
    
    print("\n" + "="*80)
    print("📈 SCÉNARIO 2: Entraînement sur BATCH_1, test sur TOUS les batches")
    print("="*80)
    
    # Entraîner sur batch_1
    batch_1_df = pd.read_csv(batch_files[0])
    X_batch1, y_batch1 = preprocess_single_batch(batch_1_df, preprocessing_objects)
    
    print(f"\n🎯 Entraînement sur {batch_files[0].stem}...")
    print(f"   📊 Taille: {len(X_batch1):,} lignes")
    
    model_batch1 = train_on_batch('batch_1', X_batch1, y_batch1)
    
    results_cross_batch = []
    
    # Tester sur tous les batches
    for batch_file in batch_files:
        batch_name = batch_file.stem
        print(f"\n🔸 Test sur {batch_name}...")
        
        batch_df = pd.read_csv(batch_file)
        X, y = preprocess_single_batch(batch_df, preprocessing_objects)
        
        metrics = evaluate_on_batch(model_batch1, X, y, batch_name)
        results_cross_batch.append(metrics)
        
        print(f"   ✅ F1-Score: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    
    df_cross_batch = pd.DataFrame(results_cross_batch)
    
    print("\n📊 RÉSULTATS - MODÈLE BATCH_1 SUR TOUS LES BATCHES:")
    print("-"*80)
    print(df_cross_batch[['batch', 'accuracy', 'f1', 'roc_auc', 'n_samples']].to_string(index=False))
    
    # ============================================================================
    # SCÉNARIO 3: Entraînement cumulatif (incremental learning)
    # ============================================================================
    
    print("\n" + "="*80)
    print("📈 SCÉNARIO 3: Entraînement CUMULATIF (batch_1, puis batch_1+2, etc.)")
    print("="*80)
    
    results_cumulative = []
    
    # Charger le batch de test
    test_df = pd.read_csv('data/raw/test/batch_test.csv')
    X_test_final, y_test_final = preprocess_single_batch(test_df, preprocessing_objects)
    
    cumulative_X = []
    cumulative_y = []
    
    for i, batch_file in enumerate(batch_files, 1):
        batch_name = batch_file.stem
        print(f"\n🔸 Ajout de {batch_name} à l'entraînement...")
        
        # Charger et prétraiter
        batch_df = pd.read_csv(batch_file)
        X, y = preprocess_single_batch(batch_df, preprocessing_objects)
        
        # Ajouter au cumulatif
        cumulative_X.append(X)
        cumulative_y.append(y)
        
        # Concaténer
        X_train_cum = pd.concat(cumulative_X, ignore_index=True)
        y_train_cum = np.concatenate(cumulative_y)
        
        print(f"   📊 Taille cumulée: {len(X_train_cum):,} lignes")
        
        # Entraîner
        model_name = f"cumulative_batch_1_to_{i}"
        model_cum = train_on_batch(model_name, X_train_cum, y_train_cum)
        
        # Évaluer sur le test set
        metrics = evaluate_on_batch(model_cum, X_test_final, y_test_final, model_name)
        metrics['batches_used'] = i
        metrics['total_samples'] = len(X_train_cum)
        results_cumulative.append(metrics)
        
        print(f"   ✅ F1-Score sur test: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    
    df_cumulative = pd.DataFrame(results_cumulative)
    
    print("\n📊 RÉSULTATS - ENTRAÎNEMENT CUMULATIF:")
    print("-"*80)
    print(df_cumulative[['batch', 'batches_used', 'total_samples', 'f1', 'roc_auc']].to_string(index=False))
    
    # ============================================================================
    # VISUALISATIONS
    # ============================================================================
    
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DES VISUALISATIONS")
    print("="*80)
    
    os.makedirs('reports/batch_analysis', exist_ok=True)
    
    # 1. Comparaison des 3 scénarios
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1-Score par scénario
    ax1 = axes[0, 0]
    x_pos = np.arange(len(batch_files))
    width = 0.25
    
    ax1.bar(x_pos - width, df_same_batch['f1'], width, label='Même batch', alpha=0.8)
    ax1.bar(x_pos, df_cross_batch['f1'], width, label='Modèle batch_1', alpha=0.8)
    ax1.bar(x_pos + width, df_cumulative['f1'][:len(batch_files)], width, label='Cumulatif', alpha=0.8)
    
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('Comparaison F1-Score par Scénario', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'B{i+1}' for i in range(len(batch_files))])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ROC-AUC par scénario
    ax2 = axes[0, 1]
    ax2.bar(x_pos - width, df_same_batch['roc_auc'], width, label='Même batch', alpha=0.8)
    ax2.bar(x_pos, df_cross_batch['roc_auc'], width, label='Modèle batch_1', alpha=0.8)
    ax2.bar(x_pos + width, df_cumulative['roc_auc'][:len(batch_files)], width, label='Cumulatif', alpha=0.8)
    
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Comparaison ROC-AUC par Scénario', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'B{i+1}' for i in range(len(batch_files))])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Évolution cumulative
    ax3 = axes[1, 0]
    ax3.plot(df_cumulative['batches_used'], df_cumulative['f1'], 
            marker='o', linewidth=2, markersize=8, label='F1-Score')
    ax3.plot(df_cumulative['batches_used'], df_cumulative['accuracy'], 
            marker='s', linewidth=2, markersize=8, label='Accuracy')
    ax3.set_xlabel('Nombre de batches utilisés')
    ax3.set_ylabel('Score')
    ax3.set_title('Évolution des performances (Entraînement Cumulatif)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df_cumulative['batches_used'])
    
    # Taille des échantillons
    ax4 = axes[1, 1]
    ax4.bar(range(1, len(batch_files)+1), 
           [r['n_samples'] for r in results_same_batch],
           alpha=0.7, color='steelblue')
    ax4.set_xlabel('Batch')
    ax4.set_ylabel('Nombre d\'échantillons')
    ax4.set_title('Taille des batches', fontweight='bold')
    ax4.set_xticks(range(1, len(batch_files)+1))
    ax4.set_xticklabels([f'B{i}' for i in range(1, len(batch_files)+1)])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('reports/batch_analysis/batch_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: reports/batch_analysis/batch_comparison.png")
    plt.close()
    
    # 2. Heatmap de détérioration des performances
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics_matrix = []
    for batch in batch_files:
        batch_name = batch.stem
        row = []
        for metric_row in results_cross_batch:
            if metric_row['batch'] == batch_name:
                row.append(metric_row['f1'])
        metrics_matrix.append(row)
    
    # Créer une matrice pour tous les modèles testés sur tous les batches
    # Pour simplifier, on montre juste la détérioration depuis batch_1
    deterioration = []
    base_score = df_cross_batch.iloc[0]['f1']  # Score sur batch_1
    for idx, row in df_cross_batch.iterrows():
        deterioration.append(row['f1'] - base_score)
    
    # Sauvegarder les résultats
    print("\n💾 Sauvegarde des résultats...")
    
    # Sauvegarder les DataFrames
    df_same_batch.to_csv('reports/batch_analysis/results_same_batch.csv', index=False)
    df_cross_batch.to_csv('reports/batch_analysis/results_cross_batch.csv', index=False)
    df_cumulative.to_csv('reports/batch_analysis/results_cumulative.csv', index=False)
    
    print("✅ results_same_batch.csv")
    print("✅ results_cross_batch.csv")
    print("✅ results_cumulative.csv")
    
    # Sauvegarder un résumé JSON
    summary = {
        'scenario_1_same_batch': {
            'mean_f1': float(df_same_batch['f1'].mean()),
            'std_f1': float(df_same_batch['f1'].std()),
            'mean_roc_auc': float(df_same_batch['roc_auc'].mean())
        },
        'scenario_2_cross_batch': {
            'batch_1_on_batch_1': float(df_cross_batch.iloc[0]['f1']),
            'batch_1_on_others_mean': float(df_cross_batch.iloc[1:]['f1'].mean()),
            'performance_drop': float(df_cross_batch.iloc[0]['f1'] - df_cross_batch.iloc[1:]['f1'].mean())
        },
        'scenario_3_cumulative': {
            'final_f1': float(df_cumulative.iloc[-1]['f1']),
            'final_roc_auc': float(df_cumulative.iloc[-1]['roc_auc']),
            'total_samples_used': int(df_cumulative.iloc[-1]['total_samples'])
        }
    }
    
    with open('reports/batch_analysis/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ summary.json")
    
    # Afficher le résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES ANALYSES")
    print("="*80)
    
    print("\n🎯 SCÉNARIO 1 (Même batch):")
    print(f"   - F1-Score moyen: {summary['scenario_1_same_batch']['mean_f1']:.4f} ± {summary['scenario_1_same_batch']['std_f1']:.4f}")
    print(f"   - ROC-AUC moyen: {summary['scenario_1_same_batch']['mean_roc_auc']:.4f}")
    
    print("\n🎯 SCÉNARIO 2 (Modèle batch_1):")
    print(f"   - Performance sur batch_1: {summary['scenario_2_cross_batch']['batch_1_on_batch_1']:.4f}")
    print(f"   - Performance moyenne sur autres: {summary['scenario_2_cross_batch']['batch_1_on_others_mean']:.4f}")
    print(f"   - Détérioration: {summary['scenario_2_cross_batch']['performance_drop']:.4f}")
    
    print("\n🎯 SCÉNARIO 3 (Cumulatif):")
    print(f"   - F1-Score final: {summary['scenario_3_cumulative']['final_f1']:.4f}")
    print(f"   - ROC-AUC final: {summary['scenario_3_cumulative']['final_roc_auc']:.4f}")
    print(f"   - Total échantillons: {summary['scenario_3_cumulative']['total_samples_used']:,}")
    
    print("\n" + "="*80)
    print("✅ ANALYSE PAR BATCH TERMINÉE!")
    print("="*80)
    print(f"\n📁 Résultats disponibles dans: reports/batch_analysis/")


if __name__ == "__main__":
    main()