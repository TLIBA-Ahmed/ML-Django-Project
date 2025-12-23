# Module de Classification des Types de Postes

## Description

Ce module implémente un système de classification pour prédire le type de poste (job title) basé sur les caractéristiques d'une offre d'emploi. Il utilise plusieurs algorithmes de machine learning et des techniques d'équilibrage des classes (SMOTE) pour améliorer les performances.

## Origine

Ce module est extrait du notebook **DS12.ipynb** qui contient une analyse complète de la classification des types de postes dans le secteur data.

## Fonctionnalités

### 1. Prétraitement des Données

Le module effectue plusieurs étapes de prétraitement :

- **Nettoyage des données** : Suppression des doublons
- **Gestion des valeurs manquantes** : Remplissage par "Unknown" pour les secteurs
- **Encodage des variables** :
  - LabelEncoder pour les colonnes catégorielles (schedule_type, sector, job_via)
  - Target Encoding pour job_title (optionnel)
- **Suppression des features problématiques** : Colonnes avec data leakage (company, location, country)

### 2. Features Utilisées

Le modèle utilise 4 features principales :

1. **job_schedule_type_enc** : Type de contrat (Full-time, Part-time, Contract, etc.)
2. **sector_enc** : Secteur d'activité de l'entreprise
3. **job_via_enc** : Plateforme de recrutement (LinkedIn, Indeed, etc.)
4. **job_skills** : Nombre de compétences requises (0-5)

### 3. Types de Postes Prédits (10 classes)

- **0** : Business Analyst
- **1** : Cloud Engineer
- **2** : Data Analyst
- **3** : Data Engineer
- **4** : Data Scientist
- **5** : Machine Learning Engineer
- **6** : Senior Data Analyst
- **7** : Senior Data Engineer
- **8** : Senior Data Scientist
- **9** : Software Engineer

### 4. Équilibrage des Classes avec SMOTE

Le dataset original présente un déséquilibre important des classes :
- Data Analyst : 31.26% (classe majoritaire)
- Machine Learning Engineer : 1.13% (classe minoritaire)

**SMOTE (Synthetic Minority Over-sampling Technique)** génère des exemples synthétiques pour équilibrer toutes les classes à 797 exemples chacune.

### 5. Modèles de Classification

Le module entraîne et compare 5 modèles :

1. **KNN (k=1)** : K-Nearest Neighbors avec recherche du meilleur k
2. **SVM (Linear)** : Support Vector Machine avec kernel linéaire
3. **Decision Tree** : Arbre de décision simple
4. **Random Forest** : Ensemble de 300 arbres (max_depth=6)
5. **XGBoost** : Gradient Boosting optimisé (300 estimators, max_depth=6)

### 6. Métriques d'Évaluation

- **Accuracy** : Taux de prédictions correctes
- **F1-Score** : Moyenne pondérée tenant compte du déséquilibre des classes
- **Confusion Matrix** : Visualisation des erreurs de classification
- **ROC Curves** : Courbes ROC pour chaque modèle

## Visualisations Disponibles

### 1. Comparaison des Modèles
Graphiques comparatifs des performances (Accuracy et F1-Score).

### 2. Matrice de Confusion
Affiche les prédictions correctes et les erreurs pour chaque classe.

### 3. Distribution des Classes (Avant/Après SMOTE)
Montre l'effet de l'équilibrage des classes.

### 4. Optimisation KNN (k-values)
F1-score en fonction du nombre de voisins pour trouver le meilleur k.

### 5. Courbes ROC
Courbes ROC pour évaluer la capacité de discrimination de chaque modèle.

### 6. Visualisation de l'Arbre de Décision
Représentation graphique de l'arbre avec les décisions prises à chaque nœud.

## Analyse du Data Leakage

Le notebook DS12.ipynb identifie plusieurs features avec data leakage :

- **job_location_enc** : 100% de leakage (supprimée)
- **job_country_enc** : 100% de leakage (supprimée)
- **company_enc** : 74.7% de leakage (supprimée)
- **job_via_enc** : 75% de leakage (conservée car acceptable)
- **job_skills** : 60% de leakage (conservée)

Ces colonnes ont été analysées et les plus problématiques ont été supprimées pour éviter que le modèle n'apprenne simplement à lire la réponse encodée.

## Performance

Les modèles atteignent des performances élevées malgré la suppression des features problématiques :

- **KNN (k=1)** : F1-Score ≈ 0.98
- **Random Forest** : F1-Score ≈ 0.97
- **XGBoost** : F1-Score ≈ 0.97

## Utilisation

### Analyse
```python
from ml_app.job_title_classification_module import JobTitleClassificationModel

# Initialiser le modèle
model = JobTitleClassificationModel()

# Charger et prétraiter les données
model.load_data()
model.preprocess_data()
model.split_data()
model.apply_smote()

# Entraîner les modèles
model.train_models()

# Évaluer les performances
results = model.evaluate_models()
print(results)

# Générer les visualisations
comparison_plot = model.visualize_comparison()
confusion_matrix_plot = model.visualize_confusion_matrix()
```

### Prédiction
```python
# Prédire le type de poste
job_data = {
    'job_schedule_type': 'Full-time',
    'sector': 'Information Technology',
    'job_via': 'LinkedIn',
    'job_skills': 3
}

predicted_title, confidence = model.predict_job_title(job_data)
print(f"Type de poste prédit : {predicted_title}")
print(f"Confiance : {confidence * 100:.2f}%")
```

## Cache des Modèles

Les modèles entraînés sont sauvegardés dans un fichier pickle pour accélérer les prédictions futures :
```
ml_app/model_cache/job_title_classification_models.pkl
```

## Intégration Django

### URLs
```python
path('job-title/', views.job_title_analysis, name='job_title_analysis'),
path('job-title/predict/', views.job_title_predict, name='job_title_predict'),
```

### Modèle Django
```python
class JobTitlePrediction(models.Model):
    job_schedule_type = models.CharField(max_length=100)
    sector = models.CharField(max_length=200)
    job_via = models.CharField(max_length=100)
    job_skills = models.IntegerField()
    predicted_job_title = models.CharField(max_length=200)
    confidence = models.FloatField(null=True, blank=True)
    model_used = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
```

## Dataset

- **Fichier** : `dataset_final3.csv`
- **Taille** : ~3190 lignes après nettoyage
- **Features** : Type de contrat, secteur, plateforme, compétences
- **Cible** : job_title_short (10 classes)

## Améliorations Futures

1. **Feature Engineering** : Ajouter de nouvelles features pertinentes
2. **Hyperparameter Tuning** : Optimiser les paramètres des modèles
3. **Ensembling** : Combiner plusieurs modèles pour améliorer les performances
4. **Cross-Validation** : Valider la robustesse des modèles
5. **Feature Importance** : Analyser l'importance de chaque feature

## Notes Techniques

### Gini Index
L'arbre de décision utilise l'indice de Gini pour choisir les meilleurs splits :
```
Gini = 1 - Σ(pi)²
```
- Gini = 0.0 : Nœud pur (une seule classe)
- Gini = 0.9 : Nœud très mélangé (plusieurs classes équilibrées)

### SMOTE
SMOTE crée de nouveaux exemples synthétiques entre les exemples existants :
1. Pour chaque exemple minoritaire
2. Trouver ses k plus proches voisins
3. Créer un nouveau point sur la ligne entre l'exemple et un voisin
4. Répéter jusqu'à équilibrer toutes les classes

## Auteur

Module extrait du notebook DS12.ipynb et intégré dans l'application Django ML.

## Licence

Ce module fait partie du projet ML Django Application.
