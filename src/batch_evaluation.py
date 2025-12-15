# -*- coding: utf-8 -*-
"""
Évaluation et Comparaison entre Batches
Analyse les résultats de modélisation de tous les batches
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
import sys
import io

# Ensure UTF-8 for stdout/stderr to avoid UnicodeEncodeError on Windows consoles
try:
    current_enc = getattr(sys.stdout, 'encoding', '') or ''
    if current_enc.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    # If anything fails, continue without raising; prints may still fail but we avoid crashing
    pass


def load_batch_results(models_dir='models'):
    """Charge les résultats de tous les batches"""
    print("="*80)
    print("EVALUATION PAR BATCH - ANALYSE COMPARATIVE")
    print("="*80)
    
    batch_dirs = sorted([d for d in Path(models_dir).iterdir() if d.is_dir() and d.name.startswith('batch_')])
    
    if not batch_dirs:
        raise FileNotFoundError(f"Aucun dossier batch trouve dans {models_dir}")
    
    print(f"\n[+] {len(batch_dirs)} batches detectes")
    
    all_results = []
    all_metadata = []
    
    for batch_dir in batch_dirs:
        batch_name = batch_dir.name
        
        # Charger comparaison
        comparison_file = batch_dir / 'models_comparison.csv'
        if comparison_file.exists():
            df = pd.read_csv(comparison_file)
            df['batch'] = batch_name
            all_results.append(df)
        
        # Charger métadonnées
        metadata_file = batch_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                all_metadata.append(metadata)
        
        print(f"  [+] {batch_name}")
    
    df_all = pd.concat(all_results, ignore_index=True)
    df_metadata = pd.DataFrame(all_metadata)
    
    return df_all, df_metadata


def create_comparison_visualizations(df_all, df_metadata, output_dir='reports/batch_analysis'):
    """Crée les visualisations comparatives"""
    print("\n" + "="*80)
    print("[*] GENERATION DES VISUALISATIONS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Comparaison F1-Score par batch et modèle
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # F1-Score
    pivot_f1 = df_all.pivot_table(values='f1', index='batch', columns=df_all.index)
    if 'Unnamed: 0' in df_all.columns:
        df_all = df_all.rename(columns={'Unnamed: 0': 'model'})
    
    pivot_f1 = df_all.pivot_table(values='f1', index='batch', columns='model')
    pivot_f1.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('F1-Score par Batch et Modèle', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
    
    # ROC-AUC
    pivot_auc = df_all.pivot_table(values='roc_auc', index='batch', columns='model')
    pivot_auc.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title('ROC-AUC par Batch et Modèle', fontweight='bold', fontsize=12)
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
    
    # Meilleurs modèles par batch
    best_models = df_metadata.sort_values('batch_name')
    axes[1, 0].barh(best_models['batch_name'], best_models['best_f1'], color='steelblue')
    axes[1, 0].set_xlabel('F1-Score')
    axes[1, 0].set_title('Performance du Meilleur Modèle par Batch', fontweight='bold', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    for i, (batch, f1) in enumerate(zip(best_models['batch_name'], best_models['best_f1'])):
        axes[1, 0].text(f1 + 0.005, i, f'{f1:.4f}', va='center', fontsize=9)
    
    # Temps d'entraînement
    pivot_time = df_all.pivot_table(values='training_time', index='batch', columns='model')
    pivot_time.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Temps d\'Entraînement par Modèle', fontweight='bold', fontsize=12)
    axes[1, 1].set_ylabel('Temps (secondes)')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_comparison.png', dpi=300, bbox_inches='tight')
    print("[+] batch_comparison.png")
    plt.close()
    
    # 2. Heatmap des performances
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Heatmap F1
    sns.heatmap(pivot_f1.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0],
                cbar_kws={'label': 'F1-Score'}, vmin=0.85, vmax=1.0)
    axes[0].set_title('Heatmap F1-Score', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Batch')
    axes[0].set_ylabel('Modèle')
    
    # Heatmap ROC-AUC
    sns.heatmap(pivot_auc.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1],
                cbar_kws={'label': 'ROC-AUC'}, vmin=0.90, vmax=1.0)
    axes[1].set_title('Heatmap ROC-AUC', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Batch')
    axes[1].set_ylabel('Modèle')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_performance.png', dpi=300, bbox_inches='tight')
    print("[+] heatmap_performance.png")
    plt.close()
    
    # 3. Distribution des performances
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot F1
    df_all.boxplot(column='f1', by='model', ax=axes[0], rot=45)
    axes[0].set_title('Distribution F1-Score par Modèle', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Modèle')
    axes[0].set_ylabel('F1-Score')
    axes[0].get_figure().suptitle('')
    
    # Boxplot ROC-AUC
    df_all.boxplot(column='roc_auc', by='model', ax=axes[1], rot=45)
    axes[1].set_title('Distribution ROC-AUC par Modèle', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Modèle')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].get_figure().suptitle('')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distribution_performance.png', dpi=300, bbox_inches='tight')
    print("[+] distribution_performance.png")
    plt.close()


def generate_summary_report(df_all, df_metadata, output_dir='reports/batch_analysis'):
    """Génère un rapport de synthèse"""
    print("\n" + "="*80)
    print("[*] GENERATION DU RAPPORT DE SYNTHESE")
    print("="*80)
    
    summary = {
        'global_stats': {
            'n_batches': int(len(df_metadata)),
            'avg_f1_across_batches': float(df_metadata['best_f1'].mean()),
            'std_f1_across_batches': float(df_metadata['best_f1'].std()),
            'best_overall_batch': str(df_metadata.loc[df_metadata['best_f1'].idxmax(), 'batch_name']),
            'best_overall_f1': float(df_metadata['best_f1'].max())
        },
        'best_models_per_batch': [
            {k: (str(v) if isinstance(v, (np.integer, np.floating)) or pd.isna(v) else v) 
             for k, v in record.items()}
            for record in df_metadata[['batch_name', 'best_model', 'best_f1', 'best_roc_auc']].to_dict('records')
        ],
        'model_consistency': {}
    }
    
    # Analyser la consistance des modèles
    model_col = 'model' if 'model' in df_all.columns else None
    
    if model_col:
        models_list = df_all[model_col].unique()
    else:
        # Si pas de colonne 'model', utiliser l'index
        df_all = df_all.reset_index()
        if 'index' in df_all.columns:
            df_all = df_all.rename(columns={'index': 'model'})
            model_col = 'model'
            models_list = df_all[model_col].unique()
        else:
            models_list = df_all.index.unique()
    
    for model in models_list:
        if model_col:
            model_data = df_all[df_all[model_col] == model]
        else:
            model_data = df_all[df_all.index == model]
        
        summary['model_consistency'][str(model)] = {
            'avg_f1': float(model_data['f1'].mean()),
            'std_f1': float(model_data['f1'].std()),
            'min_f1': float(model_data['f1'].min()),
            'max_f1': float(model_data['f1'].max()),
            'avg_roc_auc': float(model_data['roc_auc'].mean())
        }
    
    # Identifier le modèle le plus stable
    consistency_df = pd.DataFrame(summary['model_consistency']).T
    
    # Supprimer les NaN
    consistency_df = consistency_df.dropna()
    
    if len(consistency_df) == 0:
        print("[!] Aucune donnee de consistance disponible")
        summary['recommendations'] = {
            'most_stable_model': 'N/A',
            'most_stable_std': 0.0,
            'highest_avg_model': 'N/A',
            'highest_avg_f1': 0.0
        }
    else:
        most_stable_model = consistency_df['std_f1'].idxmin()
        highest_avg_model = consistency_df['avg_f1'].idxmax()
        
        summary['recommendations'] = {
            'most_stable_model': str(most_stable_model),
            'most_stable_std': float(consistency_df.loc[most_stable_model, 'std_f1']),
            'highest_avg_model': str(highest_avg_model),
            'highest_avg_f1': float(consistency_df.loc[highest_avg_model, 'avg_f1'])
        }
    
    # Sauvegarder
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Créer un rapport texte
    with open(f'{output_dir}/report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT D'ANALYSE PAR BATCH\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Nombre de batches analysés: {summary['global_stats']['n_batches']}\n")
        f.write(f"F1-Score moyen: {summary['global_stats']['avg_f1_across_batches']:.4f} ± {summary['global_stats']['std_f1_across_batches']:.4f}\n")
        f.write(f"Meilleur batch: {summary['global_stats']['best_overall_batch']} (F1={summary['global_stats']['best_overall_f1']:.4f})\n\n")
        
        f.write("MEILLEURS MODÈLES PAR BATCH:\n")
        f.write("-"*80 + "\n")
        for batch in summary['best_models_per_batch']:
            f.write(f"{batch['batch_name']:15} → {batch['best_model']:20} (F1={batch['best_f1']:.4f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMANDATIONS:\n")
        f.write("="*80 + "\n")
        f.write(f"✅ Modèle le plus STABLE: {summary['recommendations']['most_stable_model']}\n")
        f.write(f"   (Écart-type: {summary['recommendations']['most_stable_std']:.4f})\n\n")
        f.write(f"✅ Modèle avec la MEILLEURE moyenne: {summary['recommendations']['highest_avg_model']}\n")
        f.write(f"   (F1 moyen: {summary['recommendations']['highest_avg_f1']:.4f})\n")
    
    print("✅ summary.json")
    print("✅ report.txt")
    
    # Afficher le rapport
    print("\n" + "="*80)
    print("[*] RESUME DES RESULTATS")
    print("="*80)
    
    with open(f'{output_dir}/report.txt', 'r', encoding='utf-8') as f:
        print(f.read())


def main():
    """Pipeline principal d'évaluation"""
    # Charger résultats
    df_all, df_metadata = load_batch_results()
    
    # Sauvegarder données consolidées
    output_dir = 'reports/batch_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    df_all.to_csv(f'{output_dir}/all_results.csv', index=False)
    df_metadata.to_csv(f'{output_dir}/metadata_summary.csv', index=False)
    
    print("\n[+] Donnees consolidees sauvegardees")
    
    # Visualisations
    create_comparison_visualizations(df_all, df_metadata, output_dir)
    
    # Rapport
    generate_summary_report(df_all, df_metadata, output_dir)
    
    print("\n" + "="*80)
    print("[+] EVALUATION TERMINEE!")
    print("="*80)
    print(f"\n[+] Resultats disponibles dans: {output_dir}/")


if __name__ == "__main__":
    main()