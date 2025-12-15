================================================================================
MODÈLES DE PRÉDICTION - SATISFACTION PASSAGERS AÉRIENS
================================================================================

Date de création: 2025-12-15 23:10:15
Nombre de modèles: 7
Meilleur modèle: XGBoost
F1-Score du meilleur modèle: 0.9561
ROC-AUC du meilleur modèle: 0.9947

FICHIERS DISPONIBLES:
--------------------------------------------------------------------------------
- Logistic_Regression.pkl (0.00 MB)
- Random_Forest.pkl (25.21 MB)
- Bagging_Classifier.pkl (2.78 MB)
- XGBoost.pkl (0.79 MB)
- LightGBM.pkl (0.34 MB)
- MLP.pkl (0.09 MB)
- Stacking_Ensemble.pkl (4.42 MB)
- BEST_MODEL_XGBoost.pkl (copie du meilleur modèle)
- model_results_comparison.csv (tableau de comparaison)
- models_metadata.pkl (métadonnées)
- README.txt (ce fichier)

Pour charger un modèle:
  import joblib
  model = joblib.load('../PremiereEtape/models/<nom_du_modele>.pkl')
