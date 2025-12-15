# -*- coding: utf-8 -*-
"""
Modélisation - Satisfaction Passagers Aériens
Test de modèles: Bagging, Boosting, Stacking et MLP
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import joblib
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, classification_report,
                            roc_curve, auc, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration visualisation
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_params():
    """Charge les paramètres depuis params.yaml"""
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        return params.get('train', {})
    except FileNotFoundError:
        print("⚠️  params.yaml non trouvé, utilisation des paramètres par défaut")
        return {}


def load_processed_data(data_path='data/processed'):
    """Charge les données prétraitées"""
    print("="*80)
    print("✈️ MODÉLISATION - SATISFACTION PASSAGERS AÉRIENS")
    print("="*80)
    
    print("\n" + "="*80)
    print("📁 SECTION 1: CHARGEMENT DES DONNÉES PRÉTRAITÉES")
    print("="*80)
    
    X_train = pd.read_csv(f'{data_path}/X_train_processed.csv')
    y_train = pd.read_csv(f'{data_path}/y_train_processed.csv').squeeze()
    X_test = pd.read_csv(f'{data_path}/X_test_processed.csv')
    
    try:
        y_test = pd.read_csv(f'{data_path}/y_test_processed.csv').squeeze()
        has_test_target = True
    except:
        y_test = None
        has_test_target = False
    
    preprocessing_objects = joblib.load(f'{data_path}/preprocessing_objects.pkl')
    
    print("✅ Données chargées avec succès!")
    print(f"\n📊 Dimensions:")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train: {y_train.shape}")
    if has_test_target:
        print(f"   X_test: {X_test.shape}")
        print(f"   y_test: {y_test.shape}")
    
    print(f"\n🎯 Distribution des classes (y_train):")
    class_counts = y_train.value_counts()
    class_percent = y_train.value_counts(normalize=True) * 100
    
    for cls, count in class_counts.items():
        pct = class_percent[cls]
        label = preprocessing_objects['target_encoder'].inverse_transform([cls])[0]
        print(f"   Classe {cls} ({label}): {count} ({pct:.1f}%)")
    
    return X_train, X_test, y_train, y_test, has_test_target, preprocessing_objects


def create_validation_split(X_train, y_train, has_test_target, X_test, y_test, test_size=0.2):
    """Crée un ensemble de validation si nécessaire"""
    if not has_test_target:
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
        )
    else:
        X_train_final, y_train_final = X_train, y_train
        X_val, y_val = X_test, y_test
    
    return X_train_final, X_val, y_train_final, y_val


def train_baseline(X_train, y_train, X_val, y_val):
    """Entraîne le modèle baseline (Régression Logistique)"""
    print("\n" + "="*80)
    print("🎯 SECTION 2: BASELINE - RÉGRESSION LOGISTIQUE")
    print("="*80)
    
    print("🔬 Entraînement du modèle de baseline...")
    
    lr_model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear'
    )
    
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = lr_model.predict(X_val)
    y_pred_proba = lr_model.predict_proba(X_val)[:, 1]
    
    results = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'training_time': training_time
    }
    
    print("\n📊 PERFORMANCES - RÉGRESSION LOGISTIQUE:")
    print(f"   ⏱️  Temps: {results['training_time']:.2f}s")
    print(f"   📈 Accuracy: {results['accuracy']:.4f}")
    print(f"   🎯 Precision: {results['precision']:.4f}")
    print(f"   🔄 Recall: {results['recall']:.4f}")
    print(f"   ⚖️  F1-Score: {results['f1']:.4f}")
    print(f"   📊 ROC-AUC: {results['roc_auc']:.4f}")
    
    return lr_model, results, y_pred, y_pred_proba


def train_bagging_models(X_train, y_train, X_val, y_val):
    """Entraîne les modèles de Bagging"""
    print("\n" + "="*80)
    print("🌳 SECTION 3: MODÈLES DE BAGGING")
    print("="*80)
    
    print("🔬 Entraînement des modèles de Bagging...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=10, class_weight='balanced'),
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
    
    models = {
        'Random Forest': rf_model,
        'Bagging Classifier': bagging_model
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🎯 Entraînement: {name}")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'training_time': training_time
        }
        
        print(f"   ⏱️  Temps: {training_time:.2f}s")
        print(f"   📈 F1-Score: {results[name]['f1']:.4f}")
        print(f"   📊 ROC-AUC: {results[name]['roc_auc']:.4f}")
    
    return models, results


def train_boosting_models(X_train, y_train, X_val, y_val):
    """Entraîne les modèles de Boosting"""
    print("\n" + "="*80)
    print("🚀 SECTION 4: MODÈLES DE BOOSTING")
    print("="*80)
    
    try:
        import xgboost as xgb
        import lightgbm as lgb
    except ImportError:
        print("⚠️  Installation de XGBoost et LightGBM...")
        os.system('pip install xgboost lightgbm -q')
        import xgboost as xgb
        import lightgbm as lgb
    
    print("🔬 Entraînement des modèles de Boosting...")
    
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    xgb_model = xgb.XGBClassifier(
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
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbose=-1
    )
    
    models = {
        'XGBoost': xgb_model,
        'LightGBM': lgb_model
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🎯 Entraînement: {name}")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'training_time': training_time
        }
        
        print(f"   ⏱️  Temps: {training_time:.2f}s")
        print(f"   📈 F1-Score: {results[name]['f1']:.4f}")
        print(f"   📊 ROC-AUC: {results[name]['roc_auc']:.4f}")
    
    return models, results


def train_mlp(X_train, y_train, X_val, y_val):
    """Entraîne le modèle MLP"""
    print("\n" + "="*80)
    print("🧠 SECTION 5: MLP (RÉSEAU DE NEURONES)")
    print("="*80)
    
    print("🔬 Entraînement du MLP...")
    
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    start_time = time.time()
    mlp_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = mlp_model.predict(X_val)
    y_pred_proba = mlp_model.predict_proba(X_val)[:, 1]
    
    results = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'training_time': training_time
    }
    
    print("\n📊 PERFORMANCES - MLP:")
    print(f"   ⏱️  Temps: {results['training_time']:.2f}s")
    print(f"   📈 F1-Score: {results['f1']:.4f}")
    print(f"   📊 ROC-AUC: {results['roc_auc']:.4f}")
    
    return mlp_model, results, y_pred, y_pred_proba


def train_stacking(X_train, y_train, X_val, y_val):
    """Entraîne le modèle de Stacking"""
    print("\n" + "="*80)
    print("🏗️ SECTION 6: STACKING ENSEMBLE")
    print("="*80)
    
    try:
        import xgboost as xgb
        import lightgbm as lgb
    except ImportError:
        import xgboost as xgb
        import lightgbm as lgb
    
    print("🔬 Construction du modèle de Stacking...")
    
    base_models = [
        ('rf', RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )),
        ('xgb', xgb.XGBClassifier(
            n_estimators=50,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )),
        ('lgb', lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ]
    
    meta_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    start_time = time.time()
    stacking_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = stacking_model.predict(X_val)
    y_pred_proba = stacking_model.predict_proba(X_val)[:, 1]
    
    results = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'training_time': training_time
    }
    
    print("\n📊 PERFORMANCES - STACKING:")
    print(f"   ⏱️  Temps: {results['training_time']:.2f}s")
    print(f"   📈 F1-Score: {results['f1']:.4f}")
    print(f"   📊 ROC-AUC: {results['roc_auc']:.4f}")
    
    return stacking_model, results, y_pred, y_pred_proba


def compare_models(all_results):
    """Compare tous les modèles"""
    print("\n" + "="*80)
    print("📊 SECTION 7: COMPARAISON GLOBALE DES MODÈLES")
    print("="*80)
    
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df[['accuracy', 'f1', 'roc_auc', 'training_time']]
    comparison_df = comparison_df.sort_values('f1', ascending=False)
    
    print("\n📊 TABLEAU COMPARATIF:")
    print("-"*70)
    print(comparison_df.round(4))
    
    return comparison_df


def save_models(all_models, comparison_df, output_dir='models'):
    """Sauvegarde tous les modèles"""
    print("\n" + "="*80)
    print("💾 SAUVEGARDE DES MODÈLES")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🔄 Sauvegarde en cours...")
    
    for model_name, model in all_models.items():
        if model is not None:
            model_path = os.path.join(output_dir, f'{model_name.replace(" ", "_")}.pkl')
            joblib.dump(model, model_path)
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ {model_name:<25} → {file_size:.2f} MB")
    
    # Sauvegarder le meilleur modèle
    best_model_name = comparison_df.index[0]
    best_model = all_models[best_model_name]
    best_model_path = os.path.join(output_dir, f'best_model_xgboost.pkl')
    joblib.dump(best_model, best_model_path)
    
    print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
    print(f"   Sauvegardé: {best_model_path}")
    
    # Sauvegarder les résultats
    results_path = os.path.join(output_dir, 'model_results_comparison.csv')
    comparison_df.to_csv(results_path)
    print(f"\n📊 Résultats: {results_path}")
    
    # Métadonnées
    metadata = {
        'best_model': best_model_name,
        'best_f1_score': float(comparison_df.loc[best_model_name, 'f1']),
        'best_roc_auc': float(comparison_df.loc[best_model_name, 'roc_auc']),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(output_dir, 'models_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Métadonnées: {metadata_path}")


def main():
    """Pipeline principal de modélisation"""
    # Charger paramètres
    params = load_params()
    
    data_path = params.get('data_path', 'data/processed')
    output_dir = params.get('output_dir', 'models')
    
    # 1. Charger données
    X_train, X_test, y_train, y_test, has_test_target, preprocessing_objects = \
        load_processed_data(data_path)
    
    # 2. Créer validation split
    X_train_final, X_val, y_train_final, y_val = \
        create_validation_split(X_train, y_train, has_test_target, X_test, y_test)
    
    # 3. Entraîner tous les modèles
    all_models = {}
    all_results = {}
    
    # Baseline
    lr_model, lr_results, _, _ = train_baseline(X_train_final, y_train_final, X_val, y_val)
    all_models['Logistic Regression'] = lr_model
    all_results['Logistic Regression'] = lr_results
    
    # Bagging
    bagging_models, bagging_results = train_bagging_models(X_train_final, y_train_final, X_val, y_val)
    all_models.update(bagging_models)
    all_results.update(bagging_results)
    
    # Boosting
    boosting_models, boosting_results = train_boosting_models(X_train_final, y_train_final, X_val, y_val)
    all_models.update(boosting_models)
    all_results.update(boosting_results)
    
    # MLP
    mlp_model, mlp_results, _, _ = train_mlp(X_train_final, y_train_final, X_val, y_val)
    all_models['MLP'] = mlp_model
    all_results['MLP'] = mlp_results
    
    # Stacking
    stacking_model, stacking_results, _, _ = train_stacking(X_train_final, y_train_final, X_val, y_val)
    all_models['Stacking Ensemble'] = stacking_model
    all_results['Stacking Ensemble'] = stacking_results
    
    # 4. Comparer
    comparison_df = compare_models(all_results)
    
    # 5. Sauvegarder
    save_models(all_models, comparison_df, output_dir)
    
    print("\n" + "="*80)
    print("✅ MODÉLISATION TERMINÉE!")
    print("="*80)


if __name__ == "__main__":
    main()