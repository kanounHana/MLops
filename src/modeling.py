# -*- coding: utf-8 -*-
"""
Modélisation par Batch - Trouve le meilleur modèle pour CHAQUE batch
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
import time
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def load_params():
    """Charge les paramètres"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f).get('train', {})


def get_all_models(params, scale_pos_weight=1.0):
    """Retourne tous les modèles"""
    return {
        'Logistic Regression': LogisticRegression(
            random_state=params.get('random_state', 42),
            class_weight='balanced', max_iter=1000, solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=params.get('rf_n_estimators', 100),
            max_depth=params.get('rf_max_depth', 15),
            min_samples_split=params.get('rf_min_samples_split', 10),
            min_samples_leaf=params.get('rf_min_samples_leaf', 4),
            class_weight='balanced', random_state=params.get('random_state', 42), n_jobs=-1
        ),
        'Bagging Classifier': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10, class_weight='balanced'),
            n_estimators=50, random_state=params.get('random_state', 42), n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=params.get('xgb_n_estimators', 100),
            max_depth=params.get('xgb_max_depth', 8),
            learning_rate=params.get('xgb_learning_rate', 0.1),
            subsample=params.get('xgb_subsample', 0.8),
            colsample_bytree=params.get('xgb_colsample_bytree', 0.8),
            random_state=params.get('random_state', 42), n_jobs=-1,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=params.get('lgb_n_estimators', 100),
            max_depth=params.get('lgb_max_depth', 7),
            learning_rate=params.get('lgb_learning_rate', 0.1),
            num_leaves=params.get('lgb_num_leaves', 31),
            subsample=params.get('lgb_subsample', 0.8),
            colsample_bytree=params.get('lgb_colsample_bytree', 0.8),
            random_state=params.get('random_state', 42), n_jobs=-1,
            class_weight='balanced', verbose=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=tuple(params.get('mlp_hidden_layers', [64, 32])),
            activation=params.get('mlp_activation', 'relu'),
            solver='adam', alpha=params.get('mlp_alpha', 0.001),
            batch_size=params.get('mlp_batch_size', 256),
            learning_rate='adaptive', max_iter=params.get('mlp_max_iter', 200),
            random_state=params.get('random_state', 42),
            early_stopping=True, validation_fraction=0.1
        )
    }


def get_stacking_model(params):
    """Crée le modèle Stacking"""
    base_models = [
        ('rf', RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=params.get('random_state', 42),
            class_weight='balanced', n_jobs=-1
        )),
        ('xgb', xgb.XGBClassifier(
            n_estimators=50, max_depth=6, random_state=params.get('random_state', 42),
            n_jobs=-1, eval_metric='logloss'
        )),
        ('lgb', lgb.LGBMClassifier(
            n_estimators=50, max_depth=6, random_state=params.get('random_state', 42),
            n_jobs=-1, verbose=-1
        ))
    ]
    
    meta_model = LogisticRegression(
        random_state=params.get('random_state', 42),
        max_iter=1000, class_weight='balanced'
    )
    
    return StackingClassifier(
        estimators=base_models, final_estimator=meta_model,
        cv=params.get('stacking_cv', 5), n_jobs=-1
    )


def train_and_evaluate(model_name, model, X_train, y_train, X_val, y_val):
    """Entraîne et évalue un modèle"""
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'training_time': training_time
    }
    
    return model, metrics


def train_on_single_batch(batch_name, data_dir, output_dir, params):
    """Entraîne tous les modèles sur UN batch"""
    print(f"\n{'='*80}")
    print(f"[*] TRAINING: {batch_name}")
    print(f"{'='*80}")
    
    # 1. Charger données prétraitées
    X = pd.read_csv(f'{data_dir}/{batch_name}_X_processed.csv')
    y = pd.read_csv(f'{data_dir}/{batch_name}_y_processed.csv').squeeze()
    
    print(f"[+] Donnees: {X.shape[0]:,} lignes x {X.shape[1]} features")
    
    # 2. Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=params.get('validation_size', 0.2),
        random_state=params.get('random_state', 42), stratify=y
    )
    
    print(f"[+] Train: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # 3. Calculer scale_pos_weight
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # 4. Entraîner tous les modèles
    all_models = {}
    all_results = {}
    
    models = get_all_models(params, scale_pos_weight)
    
    print("\n[*] Entrainement des modeles:")
    for model_name, model in models.items():
        trained_model, metrics = train_and_evaluate(
            model_name, model, X_train, y_train, X_val, y_val
        )
        all_models[model_name] = trained_model
        all_results[model_name] = metrics
        
        print(f"  {model_name:20} -> F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f} | {metrics['training_time']:.1f}s")
    
    # 5. Stacking
    print(f"  {'Stacking Ensemble':20} -> Training...", end=" ")
    stacking = get_stacking_model(params)
    trained_stacking, stacking_metrics = train_and_evaluate(
        'Stacking Ensemble', stacking, X_train, y_train, X_val, y_val
    )
    all_models['Stacking Ensemble'] = trained_stacking
    all_results['Stacking Ensemble'] = stacking_metrics
    print(f"F1={stacking_metrics['f1']:.4f} | AUC={stacking_metrics['roc_auc']:.4f} | {stacking_metrics['training_time']:.1f}s")
    
    # 6. Trouver le meilleur
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.sort_values('f1', ascending=False)
    
    best_model_name = comparison_df.index[0]
    best_model = all_models[best_model_name]
    
    print(f"\n[+] MEILLEUR MODELE: {best_model_name}")
    print(f"   F1-Score: {comparison_df.loc[best_model_name, 'f1']:.4f}")
    print(f"   ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")
    
    # 7. Sauvegarder
    batch_output_dir = f'{output_dir}/{batch_name}'
    os.makedirs(batch_output_dir, exist_ok=True)
    
    # Sauvegarder tous les modèles
    for model_name, model in all_models.items():
        model_path = f'{batch_output_dir}/{model_name.replace(" ", "_")}.pkl'
        joblib.dump(model, model_path)
    
    # Sauvegarder le meilleur
    best_model_path = f'{batch_output_dir}/best_model.pkl'
    joblib.dump(best_model, best_model_path)
    
    # Résultats
    comparison_df.to_csv(f'{batch_output_dir}/models_comparison.csv')
    
    # Métadonnées
    metadata = {
        'batch_name': batch_name,
        'best_model': best_model_name,
        'best_f1': float(comparison_df.loc[best_model_name, 'f1']),
        'best_roc_auc': float(comparison_df.loc[best_model_name, 'roc_auc']),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    with open(f'{batch_output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[+] Sauvegarde dans: {batch_output_dir}/")
    
    return metadata


def main():
    """Entraîne sur tous les batches"""
    print("="*80)
    print("MODELISATION PAR BATCH")
    print("="*80)
    
    params = load_params()
    data_dir = params.get('data_path', 'data/processed')
    output_dir = params.get('output_dir', 'models')
    
    # Trouver tous les batches prétraités
    processed_files = sorted(Path(data_dir).glob('batch_*_X_processed.csv'))
    
    if not processed_files:
        raise FileNotFoundError(f"Aucun batch prétraité trouvé dans {data_dir}")
    
    batch_names = [f.stem.replace('_X_processed', '') for f in processed_files]
    
    print(f"\n[+] {len(batch_names)} batches a traiter")
    
    # Entraîner sur chaque batch
    all_metadata = []
    for batch_name in batch_names:
        metadata = train_on_single_batch(batch_name, data_dir, output_dir, params)
        all_metadata.append(metadata)
    
    # Résumé global
    print("\n" + "="*80)
    print("[*] RESUME GLOBAL")
    print("="*80)
    
    summary_df = pd.DataFrame(all_metadata)
    print("\n" + summary_df.to_string(index=False))
    
    # Sauvegarder résumé
    summary_df.to_csv(f'{output_dir}/global_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("[+] MODELISATION TERMINEE POUR TOUS LES BATCHES")
    print("="*80)


if __name__ == "__main__":
    main()