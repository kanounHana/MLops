# -- coding: utf-8 --
"""
Modélisation avec MLflow - Tracking des expériences
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

# Use a non-interactive backend to avoid Tkinter errors when matplotlib
# is used from worker threads or non-GUI environments (prevents
# "main thread is not in main loop" RuntimeError on Windows).
import matplotlib
matplotlib.use('Agg')

def load_params():
    """Charge les paramètres"""
    # Try UTF-8 first (common), fall back to cp1252 for Windows locales
    try:
        with open('params.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f).get('train', {})
    except UnicodeDecodeError:
        with open('params.yaml', 'r', encoding='cp1252') as f:
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


def train_and_evaluate_with_mlflow(model_name, model, X_train, y_train, X_val, y_val, batch_name, params):
    """Entraîne et évalue un modèle avec tracking MLflow"""
    
    # Démarrer un run MLflow
    with mlflow.start_run(run_name=f"{batch_name}_{model_name}"):
        
        # Logger les paramètres
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("batch_name", batch_name)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_val_samples", len(X_val))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Logger les hyperparamètres du modèle
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
        
        # Calculer les métriques
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'training_time': training_time
        }
        
        # Logger les métriques
        mlflow.log_metrics(metrics)
        
        # Logger le modèle
        if 'XGBoost' in model_name:
            mlflow.xgboost.log_model(model, "model")
        elif 'LightGBM' in model_name:
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Logger des artifacts supplémentaires
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        # Matrice de confusion
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        
        # Sauvegarder temporairement
        cm_path = f'temp_cm_{batch_name}{model_name.replace(" ", "")}.png'
        plt.savefig(cm_path)
        plt.close()
        
        # Logger l'image
        mlflow.log_artifact(cm_path)
        
        # Nettoyer
        os.remove(cm_path)
        
        # Logger des tags
        mlflow.set_tag("batch", batch_name)
        mlflow.set_tag("model_family", model_name.split()[0])
        
        print(f"  {model_name:20} -> F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f} | {training_time:.1f}s")
        
        return model, metrics, mlflow.active_run().info.run_id


def train_on_single_batch_with_mlflow(batch_name, data_dir, output_dir, params):
    """Entraîne tous les modèles sur UN batch avec MLflow"""
    
    print(f"\n{'='*80}")
    print(f"[*] TRAINING: {batch_name}")
    print(f"{'='*80}")
    
    # Définir l'expérience MLflow
    experiment_name = f"Airline_Satisfaction_{batch_name}"
    mlflow.set_experiment(experiment_name)
    
    # Charger données
    X = pd.read_csv(f'{data_dir}/{batch_name}_X_processed.csv')
    y = pd.read_csv(f'{data_dir}/{batch_name}_y_processed.csv').squeeze()
    
    print(f"[+] Donnees: {X.shape[0]:,} lignes x {X.shape[1]} features")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=params.get('validation_size', 0.2),
        random_state=params.get('random_state', 42), stratify=y
    )
    
    print(f"[+] Train: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # Scale pos weight
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # Entraîner tous les modèles
    all_models = {}
    all_results = {}
    all_run_ids = {}
    
    models = get_all_models(params, scale_pos_weight)
    
    print("\n[*] Entrainement des modeles avec MLflow:")
    
    for model_name, model in models.items():
        trained_model, metrics, run_id = train_and_evaluate_with_mlflow(
            model_name, model, X_train, y_train, X_val, y_val, batch_name, params
        )
        all_models[model_name] = trained_model
        all_results[model_name] = metrics
        all_run_ids[model_name] = run_id
    
    # Stacking
    print(f"  {'Stacking Ensemble':20} -> Training...", end=" ")
    stacking = get_stacking_model(params)
    trained_stacking, stacking_metrics, stacking_run_id = train_and_evaluate_with_mlflow(
        'Stacking Ensemble', stacking, X_train, y_train, X_val, y_val, batch_name, params
    )
    all_models['Stacking Ensemble'] = trained_stacking
    all_results['Stacking Ensemble'] = stacking_metrics
    all_run_ids['Stacking Ensemble'] = stacking_run_id
    
    print(f"F1={stacking_metrics['f1']:.4f} | AUC={stacking_metrics['roc_auc']:.4f} | {stacking_metrics['training_time']:.1f}s")
    
    # Trouver le meilleur
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.sort_values('f1', ascending=False)
    
    best_model_name = comparison_df.index[0]
    best_model = all_models[best_model_name]
    best_run_id = all_run_ids[best_model_name]
    
    print(f"\n[+] MEILLEUR MODELE: {best_model_name}")
    print(f"   F1-Score: {comparison_df.loc[best_model_name, 'f1']:.4f}")
    print(f"   ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")
    print(f"   MLflow Run ID: {best_run_id}")
    
    # Sauvegarder (comme avant)
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
        'best_roc_auc': float(comparison_df.loc[best_model_name, 'roc_auc']),
        'best_run_id': best_run_id,
        'mlflow_experiment': experiment_name,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    with open(f'{batch_output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Optionally register the best model in the MLflow Model Registry
    if params.get('register_model', False):
        try:
            registry_name = params.get('registry_name', 'AirlineSatisfaction_Production')
            model_uri = f"runs:/{best_run_id}/model"
            print(f"[+] Registering model {model_uri} as '{registry_name}' ...")
            registered_model = mlflow.register_model(model_uri=model_uri, name=registry_name)

            # Try to transition the new model version to 'Staging' (optional)
            try:
                client = MlflowClient()
                client.transition_model_version_stage(
                    name=registered_model.name,
                    version=registered_model.version,
                    stage='Staging'
                )
                print(f"[+] Model version {registered_model.version} transitioned to 'Staging'")
            except Exception as e:
                print(f"[!] Could not transition model stage: {e}")

            metadata['registered_model_name'] = registered_model.name
            metadata['registered_model_version'] = registered_model.version
        except Exception as e:
            print(f"[!] Model registry registration failed: {e}")

    print(f"\n[+] Sauvegarde dans: {batch_output_dir}/")
    print(f"[+] MLflow UI: mlflow ui --port 5000")
    
    return metadata


def main():
    """Pipeline principal avec MLflow"""
    print("="*80)
    print("MODELISATION PAR BATCH AVEC MLFLOW")
    print("="*80)
    
    # Configurer MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    
    params = load_params()
    data_dir = params.get('data_path', 'data/processed')
    output_dir = params.get('output_dir', 'models')
    
    # Trouver tous les batches
    processed_files = sorted(Path(data_dir).glob('batch_*_X_processed.csv'))
    
    if not processed_files:
        raise FileNotFoundError(f"Aucun batch pretraite trouve dans {data_dir}")
    
    batch_names = [f.stem.replace('_X_processed', '') for f in processed_files]
    
    print(f"\n[+] {len(batch_names)} batches a traiter")
    print(f"[+] MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Entraîner sur chaque batch
    all_metadata = []
    for batch_name in batch_names:
        metadata = train_on_single_batch_with_mlflow(batch_name, data_dir, output_dir, params)
        all_metadata.append(metadata)
    
    # Résumé
    print("\n" + "="*80)
    print("[*] RESUME GLOBAL")
    print("="*80)
    
    summary_df = pd.DataFrame(all_metadata)
    print("\n" + summary_df.to_string(index=False))
    
    summary_df.to_csv(f'{output_dir}/global_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("[+] MODELISATION TERMINEE POUR TOUS LES BATCHES")
    print("="*80)
    print("\n[!] Pour voir les resultats MLflow:")
    print("    mlflow ui --port 5000")
    print("    Ouvrir: http://localhost:5000")


if __name__ == "__main__":
    main()