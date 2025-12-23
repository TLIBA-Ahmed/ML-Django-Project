# Cache des Modèles ML

Ce dossier contient les modèles ML sauvegardés en format pickle pour accélérer les temps de chargement.

## Fichiers de cache

- `classification_models.pkl` - Modèles de classification (XGBoost, Random Forest, SVM, etc.)
- `clustering_models.pkl` - Modèles de clustering (K-means, PCA, etc.)
- `salary_models.pkl` - Modèles de prédiction de salaire (Linear Regression, Random Forest, etc.)

## Fonctionnement

1. **Premier chargement** : Les modèles sont entraînés et sauvegardés automatiquement
2. **Chargements suivants** : Les modèles sont chargés depuis le cache (très rapide)
3. **Réentraînement forcé** : Passer `force_retrain=True` pour recréer les modèles

## Avantages

- ⚡ Réduction du temps de chargement de 30-60 secondes à ~1 seconde
- 💾 Les modèles sont persistants entre les redémarrages du serveur
- 🔄 Rechargement automatique si les fichiers sont supprimés

## Suppression du cache

Pour forcer un réentraînement complet, supprimez simplement les fichiers `.pkl` de ce dossier.
