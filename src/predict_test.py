# -*- coding: utf-8 -*-
"""
Prédictions sur le Test Set Prétraité
Affiche la classe réelle et la classe prédite pour comparaison
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def predict_on_test(model_path, test_X_path, test_y_path, test_ids_path, output_path):
    """Fait les prédictions sur le test set"""
    
    print("="*80)
    print("🎯 PRÉDICTIONS SUR TEST SET")
    print("="*80)
    
    # 1. Charger le modèle
    print(f"\n[1] CHARGEMENT DU MODÈLE")
    print(f"    {model_path}")
    
    model = joblib.load(model_path)
    print(f"    ✅ Modèle: {type(model).__name__}")
    
    # 2. Charger le test preprocessed
    print(f"\n[2] CHARGEMENT DU TEST")
    print(f"    {test_X_path}")
    
    X_test = pd.read_csv(test_X_path)
    print(f"    ✅ {len(X_test):,} samples x {X_test.shape[1]} features")
    
    # 3. Charger les IDs
    if Path(test_ids_path).exists():
        test_ids = pd.read_csv(test_ids_path).squeeze()
        print(f"    ✅ IDs chargés")
    else:
        test_ids = pd.Series(range(len(X_test)), name='id')
        print(f"    ⚠️  IDs non trouvés, utilisation d'indices")
    
    # 4. Charger le target encoder
    print(f"\n[3] CHARGEMENT DU TARGET ENCODER")
    preprocessing_path = 'data/processed/global_preprocessing.pkl'
    preprocessing = joblib.load(preprocessing_path)
    target_encoder = preprocessing['target_encoder']
    print(f"    ✅ Classes: {target_encoder.classes_}")
    
    # 5. Charger la vraie target si disponible
    has_true_labels = False
    y_test_encoded = None
    y_test_decoded = None
    
    if Path(test_y_path).exists():
        print(f"\n[4] CHARGEMENT DES VRAIES LABELS")
        print(f"    {test_y_path}")
        
        y_test_encoded = pd.read_csv(test_y_path).squeeze()
        y_test_decoded = target_encoder.inverse_transform(y_test_encoded)
        
        has_true_labels = True
        print(f"    ✅ Vraies labels chargées")
        
        # Distribution des vraies labels
        true_dist = pd.Series(y_test_decoded).value_counts()
        print(f"\n    📊 Distribution RÉELLE:")
        for cls, count in true_dist.items():
            pct = 100 * count / len(y_test_decoded)
            print(f"       {cls:30}: {count:6,} ({pct:5.1f}%)")
    else:
        print(f"\n[4] PAS DE VRAIES LABELS")
        print(f"    ⚠️  {test_y_path} non trouvé")
        print(f"    → Mode prédiction pure (pas d'évaluation)")
    
    # 6. Prédire
    print(f"\n[5] PRÉDICTIONS")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Décoder les prédictions
    y_pred_decoded = target_encoder.inverse_transform(y_pred)
    
    print(f"    ✅ Prédictions effectuées")
    
    # 7. Créer le DataFrame de résultats
    results = pd.DataFrame({
        'id': test_ids,
        'prediction': y_pred_decoded,
        'probability_neutral_or_dissatisfied': y_pred_proba[:, 0],
        'probability_satisfied': y_pred_proba[:, 1]
    })
    
    # Ajouter la vraie classe si disponible
    if has_true_labels:
        results.insert(1, 'true_class', y_test_decoded)
        results['correct'] = (results['true_class'] == results['prediction'])
    
    # 8. Analyser les prédictions
    print(f"\n[6] ANALYSE DES PRÉDICTIONS")
    print("="*80)
    
    pred_dist = results['prediction'].value_counts()
    
    print(f"\n📊 Distribution des PRÉDICTIONS:")
    for pred, count in pred_dist.items():
        pct = 100 * count / len(results)
        bar = '█' * int(pct / 2)
        print(f"   {pred:30}: {count:6,} ({pct:5.1f}%) {bar}")
    
    # Vérifier si uniforme
    if len(pred_dist) == 1:
        print(f"\n❌ ATTENTION: Toutes les prédictions sont identiques!")
        print(f"   → Vérifier que le test a été prétraité correctement")
    elif pred_dist.max() / len(results) > 0.95:
        print(f"\n⚠️  ATTENTION: Prédictions très déséquilibrées")
        majority = pred_dist.idxmax()
        pct = 100 * pred_dist.max() / len(results)
        print(f"   Classe majoritaire: {majority} ({pct:.1f}%)")
    else:
        print(f"\n✅ Distribution semble normale")
    
    # Analyser les probabilités
    print(f"\n📊 Probabilités (satisfied):")
    probs = results['probability_satisfied']
    print(f"   Min:     {probs.min():.4f}")
    print(f"   Q1:      {probs.quantile(0.25):.4f}")
    print(f"   Médiane: {probs.median():.4f}")
    print(f"   Q3:      {probs.quantile(0.75):.4f}")
    print(f"   Max:     {probs.max():.4f}")
    
    # 9. Évaluation si vraies labels disponibles
    if has_true_labels:
        print(f"\n[7] ÉVALUATION")
        print("="*80)
        
        accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
        print(f"\n✅ Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
        
        correct_count = results['correct'].sum()
        print(f"✅ Prédictions correctes: {correct_count:,} / {len(results):,}")
        
        # F1-scores
        from sklearn.metrics import precision_score, recall_score
        
        # Moyenne
        f1 = f1_score(y_test_decoded, y_pred_decoded, average='macro')
        precision = precision_score(y_test_decoded, y_pred_decoded, average='macro')
        recall = recall_score(y_test_decoded, y_pred_decoded, average='macro')
        
        print(f"\n📈 Métriques Globales:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # Par classe
        f1_per_class = f1_score(y_test_decoded, y_pred_decoded, average=None, labels=target_encoder.classes_)
        precision_per_class = precision_score(y_test_decoded, y_pred_decoded, average=None, labels=target_encoder.classes_, zero_division=0)
        recall_per_class = recall_score(y_test_decoded, y_pred_decoded, average=None, labels=target_encoder.classes_, zero_division=0)
        
        print(f"\n📈 Métriques par Classe:")
        for i, cls in enumerate(target_encoder.classes_):
            print(f"\n   {cls}:")
            print(f"      Precision: {precision_per_class[i]:.4f}")
            print(f"      Recall:    {recall_per_class[i]:.4f}")
            print(f"      F1-Score:  {f1_per_class[i]:.4f}")
        
        # Confusion Matrix
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=target_encoder.classes_)
        
        # Afficher de manière lisible
        cm_df = pd.DataFrame(
            cm, 
            index=[f"True: {cls}" for cls in target_encoder.classes_],
            columns=[f"Pred: {cls}" for cls in target_encoder.classes_]
        )
        print(f"\n{cm_df}")
        
        # Classification Report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test_decoded, y_pred_decoded, digits=4))
        
        # Analyser les erreurs
        errors = results[~results['correct']]
        if len(errors) > 0:
            print(f"\n⚠️  ANALYSE DES ERREURS ({len(errors):,} erreurs):")
            
            error_types = errors.groupby(['true_class', 'prediction']).size().reset_index(name='count')
            error_types = error_types.sort_values('count', ascending=False)
            
            print(f"\n   Types d'erreurs:")
            for _, row in error_types.iterrows():
                pct = 100 * row['count'] / len(errors)
                print(f"      {row['true_class']:30} → {row['prediction']:30}: {row['count']:4,} ({pct:5.1f}%)")
    
    # 10. Sauvegarder
    results.to_csv(output_path, index=False)
    
    print(f"\n[8] SAUVEGARDE")
    print("="*80)
    print(f"    ✅ {output_path}")
    print(f"    ✅ {len(results):,} prédictions")
    
    if has_true_labels:
        print(f"\n    Colonnes sauvegardées:")
        print(f"      - id: Identifiant")
        print(f"      - true_class: Vraie classe ✅")
        print(f"      - prediction: Classe prédite")
        print(f"      - correct: Prédiction correcte (True/False)")
        print(f"      - probability_neutral_or_dissatisfied")
        print(f"      - probability_satisfied")
    
    return results


def main():
    """Pipeline de prédiction"""
    
    print("\n" + "🎯"*40)
    print("PRÉDICTIONS SUR TEST SET AVEC COMPARAISON")
    print("🎯"*40 + "\n")
    
    # Chemins
    model_path = 'models/batch_1/best_model.pkl'  # Ou choisir un autre batch
    test_X_path = 'data/processed/batch_test_X_processed.csv'
    test_y_path = 'data/processed/batch_test_y_processed.csv'  # Si disponible
    test_ids_path = 'data/processed/batch_test_ids.csv'
    output_path = 'test_predictions.csv'
    
    # Vérifications
    if not Path(test_X_path).exists():
        print(f"❌ Test preprocessed non trouvé: {test_X_path}")
        print(f"\n💡 Exécutez d'abord:")
        print(f"   python preprocess_test.py")
        return
    
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        print(f"\n💡 Modèles disponibles:")
        for batch_dir in Path('models').glob('batch_*'):
            best_model = batch_dir / 'best_model.pkl'
            if best_model.exists():
                metadata_path = batch_dir / 'metadata.json'
                if metadata_path.exists():
                    import json
                    with open(metadata_path) as f:
                        meta = json.load(f)
                    print(f"   - {best_model} (F1={meta['best_f1']:.4f})")
                else:
                    print(f"   - {best_model}")
        return
    
    # Prédire
    results = predict_on_test(model_path, test_X_path, test_y_path, test_ids_path, output_path)
    
    # Afficher un échantillon
    print(f"\n" + "="*80)
    print("📄 ÉCHANTILLON DES PRÉDICTIONS")
    print("="*80)
    
    if 'true_class' in results.columns:
        # Afficher avec comparaison
        display_cols = ['id', 'true_class', 'prediction', 'correct', 'probability_satisfied']
        print(results[display_cols].head(30).to_string(index=False))
        
        # Afficher quelques erreurs si disponibles
        errors = results[~results['correct']]
        if len(errors) > 0:
            print(f"\n" + "="*80)
            print(f"⚠️  ÉCHANTILLON DES ERREURS ({len(errors):,} total)")
            print("="*80)
            print(errors[display_cols].head(20).to_string(index=False))
    else:
        # Afficher sans comparaison
        display_cols = ['id', 'prediction', 'probability_satisfied']
        print(results[display_cols].head(30).to_string(index=False))
    
    print(f"\n" + "="*80)
    print("✅ PRÉDICTIONS TERMINÉES")
    print("="*80)
    print(f"\nFichier de sortie: {output_path}")
    
    if 'true_class' in results.columns:
        accuracy = (results['correct'].sum() / len(results)) * 100
        print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()