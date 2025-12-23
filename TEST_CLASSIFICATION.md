# Guide de Test - Classification des Plateformes

## Accès rapide
- **URL d'analyse** : http://127.0.0.1:8000/classification/
- **URL de prédiction** : http://127.0.0.1:8000/classification/predict/

## Fonctionnalités testées

### ✅ Module de Classification
- [x] `classification_module.py` créé avec classe `PlatformClassificationModel`
- [x] Chargement et prétraitement des données jobs_data.csv
- [x] Entraînement de 5 modèles : XGBoost, Random Forest, KNN, SVM, Decision Tree
- [x] Méthodes d'évaluation et visualisation

### ✅ Modèle Django
- [x] `PlatformPrediction` ajouté dans models.py
- [x] Migration 0002_platformprediction créée et appliquée
- [x] Modèle enregistré dans admin.py

### ✅ Vues
- [x] `classification_analysis()` - Analyse comparative des modèles
- [x] `classification_predict()` - Prédiction avec formulaire
- [x] Dropdowns dynamiques depuis le dataset
- [x] Sauvegarde des prédictions en base

### ✅ Templates
- [x] `classification_analysis.html` - Affichage des résultats et graphiques
- [x] `classification_predict.html` - Formulaire avec Select2
- [x] `home.html` - Carte de classification ajoutée
- [x] `history.html` - Section prédictions de plateforme

### ✅ Configuration
- [x] URLs ajoutées dans urls.py
- [x] XGBoost installé (v3.1.2)
- [x] requirements.txt mis à jour
- [x] README.md mis à jour

## Tests à effectuer

### 1. Page d'analyse
```
http://127.0.0.1:8000/classification/
```
**À vérifier** :
- [ ] Graphiques de comparaison des modèles s'affichent
- [ ] Tableau avec les métriques (accuracy, overfitting, CV score)
- [ ] Matrice de confusion du meilleur modèle
- [ ] Indication du meilleur modèle (XGBoost attendu)

### 2. Page de prédiction
```
http://127.0.0.1:8000/classification/predict/
```
**À vérifier** :
- [ ] Formulaire avec tous les champs requis
- [ ] Dropdowns pour job_title, country, company_name
- [ ] Select2 activé pour les longues listes
- [ ] Options binaires pour work_from_home, no_degree, health_insurance
- [ ] Choix du modèle (avec option "Auto")

### 3. Faire une prédiction
**Exemple de données** :
- Titre : "Data Scientist" (ou "Senior Data Engineer")
- Pays : "United States" (ou autre pays disponible)
- Entreprise : Choisir dans la liste
- Travail à distance : Oui/Non
- Pas de diplôme : Oui/Non
- Assurance santé : Oui/Non
- Modèle : Laisser vide pour auto

**Résultat attendu** :
- [ ] Plateforme prédite affichée (ex: LinkedIn, BeBee, Jooble)
- [ ] Probabilités pour toutes les plateformes
- [ ] Barres de progression avec pourcentages
- [ ] Détails du job affichés
- [ ] Titre simplifié calculé (ex: "Senior Data Scientist")

### 4. Historique
```
http://127.0.0.1:8000/history/
```
**À vérifier** :
- [ ] Section "Prédictions de Plateforme" visible
- [ ] Carte statistique avec le compte
- [ ] Tableau avec les prédictions sauvegardées
- [ ] Colonnes : date, job simplifié, pays, entreprise, plateforme, confiance, modèle

### 5. Page d'accueil
```
http://127.0.0.1:8000/
```
**À vérifier** :
- [ ] Carte "Classification des Plateformes" ajoutée
- [ ] Boutons "Voir l'Analyse" et "Prédire une Plateforme"
- [ ] Statistiques mises à jour : 3 notebooks, 13+ modèles, 15+ visualisations
- [ ] Section technologies mise à jour avec classification

## Données nécessaires

Le fichier `jobs_data.csv` doit être présent dans le dossier parent du projet ou être téléchargé automatiquement.

**Colonnes utilisées** :
- job_title
- job_via (cible - plateforme)
- job_country
- company_name
- job_work_from_home (binaire)
- job_no_degree_mention (binaire)
- job_health_insurance (binaire)

## Performance attendue

**Meilleurs modèles** (basé sur le notebook) :
1. **XGBoost** : ~83.7% accuracy
2. **Random Forest** : ~77% accuracy
3. **KNN** : ~70% accuracy
4. **SVM** : ~71% accuracy
5. **Decision Tree** : ~65% accuracy

## Dépannage

### Erreur : Module 'xgboost' not found
```powershell
.\venv\Scripts\python.exe -m pip install xgboost
```

### Erreur : Table 'PlatformPrediction' doesn't exist
```powershell
python manage.py makemigrations
python manage.py migrate
```

### Erreur : jobs_data.csv not found
Vérifier que le fichier est dans le bon emplacement ou ajuster le chemin dans `classification_module.py`

### Lenteur au chargement
Premier chargement peut être long car :
- Prétraitement des données (~150k lignes)
- Équilibrage des classes (15000 échantillons par plateforme)
- Entraînement de 5 modèles
- Cross-validation

**Solution** : Considérer la mise en cache des modèles pour la production

## Notes

- Les prédictions sont sauvegardées dans la base SQLite
- Les modèles sont ré-entraînés à chaque requête (à optimiser pour la production)
- Select2 améliore l'expérience utilisateur pour les longues listes
- Les visualisations sont générées en base64 pour l'affichage dans les templates
