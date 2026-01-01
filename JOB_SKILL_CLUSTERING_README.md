# Module de Clustering par Compétences (Job Skill Clustering)

## Vue d'ensemble

Ce module analyse les jobs basés sur leurs compétences techniques requises et les regroupe en clusters distincts représentant différents profils professionnels dans le domaine de la data.

## Origine

Ce module est extrait et adapté du notebook `Clustering.ipynb` qui effectue une analyse approfondie de clustering sur les compétences des jobs.

## Dataset

Le module utilise deux fichiers de données:
- **processed_data_jobs.csv**: Matrice binaire des compétences (chaque colonne = une compétence, 0 ou 1)
- **df_final_ml.csv**: Dataset complet avec informations supplémentaires (titres de jobs, etc.)

## Méthodologie

### 1. Prétraitement
- Standardisation des features de compétences binaires
- Application de StandardScaler pour normaliser les données

### 2. Détermination du nombre optimal de clusters
- **Méthode du Coude**: Analyse de l'inertie pour K de 2 à 10
- **Score de Silhouette**: Mesure de la qualité de la séparation des clusters
- **Résultat optimal**: K=3 clusters

### 3. Algorithme de clustering
- **K-Means** avec k=3
- 10 initialisations aléatoires (n_init=10)
- Maximum 300 itérations
- Random state = 42 pour la reproductibilité

### 4. Visualisation
- **PCA (2 composantes)**: Visualisation des clusters dans un espace 2D
- **Distribution**: Taille et répartition des clusters
- **Top Compétences**: Les 10 compétences les plus demandées par cluster
- **Radar Chart**: Profil comparatif des compétences entre clusters
- **Heatmap**: Matrice de présence des compétences par cluster

## Les 3 Clusters Identifiés

### Cluster 0: Business / BI Analysts
**Caractéristiques:**
- Orientation analyse business et reporting
- Focus sur la visualisation de données
- Outils décisionnels

**Top Compétences:**
- Excel
- Tableau
- Power BI
- SQL
- Python (niveau intermédiaire)

**Jobs typiques:**
- Business Analyst
- Data Analyst
- BI Analyst

### Cluster 1: Senior Data Engineers (Big Data & Cloud)
**Caractéristiques:**
- Expertise en ingénierie de données
- Infrastructure Big Data et Cloud
- Pipelines de données à grande échelle

**Top Compétences:**
- Spark
- AWS
- Azure
- Hadoop
- Kafka
- Kubernetes
- Docker
- Airflow

**Jobs typiques:**
- Data Engineer
- Senior Data Engineer
- Big Data Engineer
- Cloud Data Engineer

### Cluster 2: Data Scientists & Applied Analysts
**Caractéristiques:**
- Modélisation statistique et ML
- Analyse appliquée
- Recherche et développement

**Top Compétences:**
- Python
- R
- SQL
- Machine Learning libraries
- Jupyter
- Git

**Jobs typiques:**
- Data Scientist
- Senior Data Scientist
- ML Engineer
- Research Analyst

## Métriques de Performance

Le modèle utilise plusieurs métriques pour évaluer la qualité du clustering:
- **Inertie**: Mesure de la compacité des clusters
- **Davies-Bouldin Score**: Plus bas = meilleure séparation (optimal proche de 0)
- **Calinski-Harabasz Score**: Plus haut = meilleurs clusters définis

## Fonctionnalités de l'Application

### 1. Analyse (Admin uniquement)
URL: `/job-skill-clustering/`

Visualisations disponibles:
- Méthode du coude
- Score de silhouette
- Visualisation PCA avec centroïdes
- Distribution des clusters
- Histogramme du nombre de compétences
- Top 10 compétences par cluster
- Comparaison des compétences clés
- Radar chart des profils
- Distribution des titres de jobs
- Heatmap des compétences

### 2. Prédiction (Utilisateurs authentifiés)
URL: `/job-skill-clustering/predict/`

Fonctionnalités:
- Sélection de compétences via checkboxes
- Prédiction du cluster correspondant
- Affichage du profil professionnel
- Top compétences du cluster identifié
- Jobs typiques associés

## Structure du Code

### Module Principal
`job_skill_clustering_module.py`

Classes et méthodes principales:
```python
class JobSkillClusteringModel:
    - load_data()
    - preprocess_data()
    - perform_pca()
    - elbow_method()
    - silhouette_analysis()
    - perform_kmeans()
    - visualize_pca_clusters()
    - get_top_skills_by_cluster()
    - predict_cluster()
```

### Vues Django
`views.py`

- `job_skill_clustering_analysis()`: Page d'analyse complète
- `job_skill_clustering_predict()`: Page de prédiction interactive

### Templates
- `job_skill_clustering_analysis.html`: Affichage des analyses
- `job_skill_clustering_predict.html`: Interface de prédiction

## Cache et Performance

Le modèle utilise un système de cache pour optimiser les performances:
- Les modèles entraînés (K-means, PCA, Scaler) sont sauvegardés dans `model_cache/`
- Chargement automatique depuis le cache si disponible
- Force retrain avec paramètre `force_retrain=True`

## Utilisation

### Pour les Administrateurs
1. Accéder à la page d'analyse
2. Visualiser toutes les analyses de clustering
3. Comprendre les profils de compétences
4. Identifier les tendances du marché

### Pour les Utilisateurs
1. Accéder à la page de prédiction
2. Sélectionner les compétences possédées ou recherchées
3. Obtenir le profil professionnel correspondant
4. Découvrir les compétences complémentaires à acquérir

## Dépendances

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
pickle
```

## Améliorations Futures

- [ ] Ajout de DBSCAN pour comparaison
- [ ] Clustering hiérarchique avec dendrogramme
- [ ] Analyse de l'évolution des compétences dans le temps
- [ ] Recommandations de compétences à acquérir
- [ ] Export des résultats en PDF/CSV
- [ ] Intégration avec le chatbot pour recommandations personnalisées

## Notes Techniques

### Pourquoi K=3 ?
L'analyse a montré qu'avec K=5, plusieurs clusters présentaient une forte similarité dans leur composition de compétences (overlap de Python, SQL, outils cloud). K=3 offre:
- Meilleure interprétabilité
- Séparation plus nette des profils
- Représentation plus fidèle des archétypes professionnels

### Différence avec clustering_module.py
- `clustering_module.py`: Clustering général sur `ai_job_dataset.csv` (salaires, expérience, localisation)
- `job_skill_clustering_module.py`: Clustering spécifique sur les compétences techniques (matrice binaire)

## Références

- Notebook source: `Clustering.ipynb`
- Dataset: `processed_data_jobs.csv`, `df_final_ml.csv`
- Documentation K-means: https://scikit-learn.org/stable/modules/clustering.html#k-means
