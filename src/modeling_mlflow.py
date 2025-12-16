# -*- coding: utf-8 -*-
"""
Modélisation avec MLflow - Gestion du déséquilibre avec class_weight='balanced'
Entraîne sur CHAQUE batch individuellement
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report, 
                            confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')


def load_params():
    """Charge les paramètres"""
    try:
        with open('params.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f).get('train', {})
    except UnicodeDecodeError:
        with open('params.yaml', 'r', encoding='cp1252') as f:
            return yaml.safe_load(f).get('train', {})


def get_all_models_balanced(params, scale_pos_weight=1.0):
    """Retourne tous les modèles avec class_weight='balanced'"""
    return {
        'Logistic Regression': LogisticRegression(
            random_state=params.get('random_state', 42),
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear',
            C=params.get('lr_C', 1.0)
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=params.get('rf_n_estimators', 150),
            max_depth=params.get('rf_max_depth', 20),
            min_samples_split=params.get('rf_min_samples_split', 5),
            min_samples_leaf=params.get('rf_min_samples_leaf', 2),
            class_weight='balanced',
            random_state=params.get('random_state', 42),
            n_jobs=-1,
            max_features='sqrt'
        ),
        'Bagging Classifier': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=12, class_weight='balanced'),
            n_estimators=50,
            random_state=params.get('random_state', 42),
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=params.get('xgb_n_estimators', 150),
            max_depth=params.get('xgb_max_depth', 8),
            learning_rate=params.get('xgb_learning_rate', 0.05),
            subsample=params.get('xgb_subsample', 0.8),
            colsample_bytree=params.get('xgb_colsample_bytree', 0.8),
            random_state=params.get('random_state', 42),
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            min_child_weight=3,
            gamma=0.1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=params.get('lgb_n_estimators', 150),
            max_depth=params.get('lgb_max_depth', 8),
            learning_rate=params.get('lgb_learning_rate', 0.05),
            num_leaves=params.get('lgb_num_leaves', 50),
            subsample=params.get('lgb_subsample', 0.8),
            colsample_bytree=params.get('lgb_colsample_bytree', 0.8),
            random_state=params.get('random_state', 42),
            n_jobs=-1,
            class_weight='balanced',
            verbose=-1,
            min_child_samples=20
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=tuple(params.get('mlp_hidden_layers', [128, 64, 32])),
            activation=params.get('mlp_activation', 'relu'),
            solver='adam',
            alpha=params.get('mlp_alpha', 0.0001),
            batch_size=params.get('mlp_batch_size', 256),
            learning_rate='adaptive',
            max_iter=params.get('mlp_max_iter', 300),
            random_state=params.get('random_state', 42),
            early_stopping=True,
            validation_fraction=0.1
        )
    }


def get_stacking_model_balanced(params, scale_pos_weight=1.0):
    """Crée le modèle Stacking avec class_weight"""
    base_models = [
        ('rf', RandomForestClassifier(
            n_estimators=70,
            max_depth=15,
            random_state=params.get('random_state', 42),
            class_weight='balanced',
            n_jobs=-1
        )),
        ('xgb', xgb.XGBClassifier(
            n_estimators=70,
            max_depth=7,
            random_state=params.get('random_state', 42),
            n_jobs=-1,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )),
        ('lgb', lgb.LGBMClassifier(
            n_estimators=70,
            max_depth=7,
            random_state=params.get('random_state', 42),
            n_jobs=-1,
            verbose=-1,
            class_weight='balanced'
        ))
    ]
    
    meta_model = LogisticRegression(
        random_state=params.get('random_state', 42),
        max_iter=1000,
        class_weight='balanced'
    )
    
    return StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=params.get('stacking_cv', 5),
        n_jobs=-1
    )


def train_and_evaluate_with_mlflow(model_name, model, X_train, y_train, X_val, y_val, 
                                   batch_name, params):
    """Entraîne et évalue avec métriques complètes"""
    
    with mlflow.start_run(run_name=f"{batch_name}_{model_name}"):
        
        # Logger paramètres
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("batch_name", batch_name)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_val_samples", len(X_val))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Distribution
        train_dist = pd.Series(y_train).value_counts()
        mlflow.log_param("train_class_0", int(train_dist.get(0, 0)))
        mlflow.log_param("train_class_1", int(train_dist.get(1, 0)))
        
        # Hyperparamètres
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
            for key, value in model_params.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(f"model_{key}", value)
        
        # Entraînement
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Prédictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Métriques globales
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'training_time': training_time
        }
        
        # Métriques par classe
        precision_per_class = precision_score(y_val, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_val, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
        
        metrics['precision_class_0'] = float(precision_per_class[0])
        metrics['precision_class_1'] = float(precision_per_class[1])
        metrics['recall_class_0'] = float(recall_per_class[0])
        metrics['recall_class_1'] = float(recall_per_class[1])
        metrics['f1_class_0'] = float(f1_per_class[0])
        metrics['f1_class_1'] = float(f1_per_class[1])
        
        mlflow.log_metrics(metrics)
        
        # Logger le modèle
        if 'XGBoost' in model_name:
            mlflow.xgboost.log_model(model, "model")
        elif 'LightGBM' in model_name:
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Confusion Matrix
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        
        cm_path = f'temp_cm_{batch_name}_{model_name.replace(" ", "")}.png'
        plt.savefig(cm_path)
        plt.close()
        
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        
        # Classification Report
        report_path = f'temp_report_{batch_name}_{model_name.replace(" ", "")}.txt'
        with open(report_path, 'w') as f:
            f.write(classification_report(y_val, y_pred))
        mlflow.log_artifact(report_path)
        os.remove(report_path)
        
        # Tags
        mlflow.set_tag("batch", batch_name)
        mlflow.set_tag("model_family", model_name.split()[0])
        
        print(f"  {model_name:20} -> F1={metrics['f1']:.4f} | "
              f"F1_C0={metrics['f1_class_0']:.4f} | F1_C1={metrics['f1_class_1']:.4f} | "
              f"AUC={metrics['roc_auc']:.4f}")
        
        return model, metrics, mlflow.active_run().info.run_id


def train_on_single_batch_with_mlflow(batch_name, data_dir, output_dir, params):
    """Entraîne tous les modèles sur UN batch"""
    
    print(f"\n{'='*80}")
    print(f"[*] TRAINING: {batch_name}")
    print(f"{'='*80}")
    
    experiment_name = f"Airline_Satisfaction_{batch_name}"
    mlflow.set_experiment(experiment_name)
    
    # Charger données
    X = pd.read_csv(f'{data_dir}/{batch_name}_X_processed.csv')
    y = pd.read_csv(f'{data_dir}/{batch_name}_y_processed.csv').squeeze()
    
    print(f"[+] Données: {X.shape[0]:,} lignes x {X.shape[1]} features")
    
    # Distribution
    class_dist = y.value_counts()
    print(f"\n[*] DISTRIBUTION:")
    for cls, count in class_dist.items():
        pct = 100 * count / len(y)
        print(f"   Classe {cls}: {count:,} ({pct:.1f}%)")
    
    # Split stratifié
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=params.get('validation_size', 0.2),
        random_state=params.get('random_state', 42),
        stratify=y
    )
    
    print(f"\n[+] Train: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # Scale pos weight pour XGBoost
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    print(f"[+] Scale pos weight: {scale_pos_weight:.2f}")
    
    # Entraîner tous les modèles
    all_models = {}
    all_results = {}
    all_run_ids = {}
    
    models = get_all_models_balanced(params, scale_pos_weight)
    
    print("\n[*] Entraînement des modèles:")
    
    for model_name, model in models.items():
        trained_model, metrics, run_id = train_and_evaluate_with_mlflow(
            model_name, model, X_train, y_train, X_val, y_val, batch_name, params
        )
        all_models[model_name] = trained_model
        all_results[model_name] = metrics
        all_run_ids[model_name] = run_id
    
    # Stacking
    print(f"\n  {'Stacking Ensemble':20} -> Training...")
    stacking = get_stacking_model_balanced(params, scale_pos_weight)
    trained_stacking, stacking_metrics, stacking_run_id = train_and_evaluate_with_mlflow(
        'Stacking Ensemble', stacking, X_train, y_train, X_val, y_val, batch_name, params
    )
    all_models['Stacking Ensemble'] = trained_stacking
    all_results['Stacking Ensemble'] = stacking_metrics
    all_run_ids['Stacking Ensemble'] = stacking_run_id
    
    # Comparaison
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.sort_values('f1', ascending=False)
    
    best_model_name = comparison_df.index[0]
    best_model = all_models[best_model_name]
    best_run_id = all_run_ids[best_model_name]
    
    print(f"\n{'='*80}")
    print(f"[+] MEILLEUR MODÈLE: {best_model_name}")
    print(f"{'='*80}")
    print(f"   F1-Score Global: {comparison_df.loc[best_model_name, 'f1']:.4f}")
    print(f"   F1-Score Classe 0: {comparison_df.loc[best_model_name, 'f1_class_0']:.4f}")
    print(f"   F1-Score Classe 1: {comparison_df.loc[best_model_name, 'f1_class_1']:.4f}")
    print(f"   ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")
    print(f"   MLflow Run ID: {best_run_id}")
    
    # Sauvegarder
    batch_output_dir = f'{output_dir}/{batch_name}'
    os.makedirs(batch_output_dir, exist_ok=True)
    
    for model_name, model in all_models.items():
        model_path = f'{batch_output_dir}/{model_name.replace(" ", "_")}.pkl'
        joblib.dump(model, model_path)
    
    best_model_path = f'{batch_output_dir}/best_model.pkl'
    joblib.dump(best_model, best_model_path)
    
    comparison_df.to_csv(f'{batch_output_dir}/models_comparison.csv')
    
    # Métadonnées
    import json
    metadata = {
        'batch_name': batch_name,
        'best_model': best_model_name,
        'best_f1': float(comparison_df.loc[best_model_name, 'f1']),
        'best_f1_class_0': float(comparison_df.loc[best_model_name, 'f1_class_0']),
        'best_f1_class_1': float(comparison_df.loc[best_model_name, 'f1_class_1']),
        'best_roc_auc': float(comparison_df.loc[best_model_name, 'roc_auc']),
        'best_run_id': best_run_id,
        'mlflow_experiment': experiment_name,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    with open(f'{batch_output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Model Registry (optionnel)
    if params.get('register_model', False):
        try:
            registry_name = params.get('registry_name', 'AirlineSatisfaction_Production')
            model_uri = f"runs:/{best_run_id}/model"
            print(f"\n[+] Registering model as '{registry_name}'...")
            registered_model = mlflow.register_model(model_uri=model_uri, name=registry_name)
            
            try:
                client = MlflowClient()
                client.transition_model_version_stage(
                    name=registered_model.name,
                    version=registered_model.version,
                    stage='Staging'
                )
                print(f"[+] Model version {registered_model.version} → 'Staging'")
            except Exception as e:
                print(f"[!] Could not transition stage: {e}")
            
            metadata['registered_model_name'] = registered_model.name
            metadata['registered_model_version'] = registered_model.version
            
            with open(f'{batch_output_dir}/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"[!] Model registry failed: {e}")
    
    print(f"\n[+] Sauvegarde dans: {batch_output_dir}/")
    
    return metadata


def main():
    """Pipeline principal"""
    print("="*80)
    print("MODELISATION PAR BATCH AVEC GESTION DU DÉSÉQUILIBRE")
    print("="*80)
    
    mlflow.set_tracking_uri("file:./mlruns")
    
    params = load_params()
    data_dir = params.get('data_path', 'data/processed')
    output_dir = params.get('output_dir', 'models')
    
    processed_files = sorted(Path(data_dir).glob('batch_*_X_processed.csv'))
    
    if not processed_files:
        raise FileNotFoundError(f"Aucun batch prétraité trouvé dans {data_dir}")
    
    batch_names = [f.stem.replace('_X_processed', '') for f in processed_files]
    
    print(f"\n[+] {len(batch_names)} batches à traiter")
    print(f"[+] MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Entraîner sur chaque batch
    all_metadata = []
    for batch_name in batch_names:
        metadata = train_on_single_batch_with_mlflow(batch_name, data_dir, output_dir, params)
        all_metadata.append(metadata)
    
    # Résumé
    print("\n" + "="*80)
    print("[*] RÉSUMÉ GLOBAL")
    print("="*80)
    
    summary_df = pd.DataFrame(all_metadata)
    print("\n" + summary_df.to_string(index=False))
    
    summary_df.to_csv(f'{output_dir}/global_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("[+] MODÉLISATION TERMINÉE")
    print("="*80)
    print("\n[!] Pour voir les résultats MLflow:")
    print("    mlflow ui --port 5000")
    print("    http://localhost:5000")


if __name__ == "__main__":
    main()