# 🤖 Application Django Machine Learning - IA Job Market

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.0-092E20?style=for-the-badge&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

Cette application web développée avec **Django** permet d'explorer, d'analyser et de prédire des données liées au marché de l'emploi en Intelligence Artificielle à travers trois modules de Machine Learning distincts.

---

## 🌟 Fonctionnalités Principales

### 1. 🧩 Clustering des Jobs IA
* **Analyse interactive** : Visualisation de la méthode du coude et réduction de dimension (PCA).
* **Modélisation** : Utilisation de K-Means pour segmenter le marché.
* **Prédiction** : Interface pour attribuer un cluster à un nouveau poste.

### 2. 💰 Prédiction de Salaire (Régression)
* **Multi-modèles** : Comparaison entre Régression Linéaire, Polynomiale, Arbre de Décision, Random Forest et Gradient Boosting.
* **Visualisation** : Matrice de corrélation et graphiques de performance.
* **Estimation** : Formulaire permettant d'estimer un salaire selon le profil.

### 3. 🏢 Classification des Plateformes
* **Performance** : Modèle **XGBoost** atteignant environ **83.7%** d'accuracy.
* **Optimisation** : Analyse des probabilités pour chaque plateforme de recrutement (LinkedIn, Indeed, etc.).
* **Aide à la décision** : Recommandation de la meilleure plateforme pour poster une offre.

### 4. 📜 Historique & Suivi
* Sauvegarde systématique des prédictions dans une base de données **SQLite**.
* Consultation et filtrage des anciennes analyses.

---

## 🚀 Installation et Configuration

### Prérequis
* Python 3.10 ou +
* `pip` (gestionnaire de paquets)

### Étapes d'installation

1.  **Cloner le projet**
    ```bash
    git clone [https://github.com/TLIBA-Ahmed/django-ml-app.git](https://github.com/TLIBA-Ahmed/django-ml-app.git)
    cd django-ml-app
    ```

2.  **Créer un environnement virtuel**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\Activate.ps1

    # Linux / MacOS
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialiser la base de données**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

5.  **Lancer l'application**
    ```bash
    python manage.py runserver
    ```
    L'application est maintenant disponible sur [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 📂 Structure du Projet

```text
ml_django_project/
├── ml_app/                 # Application principale
│   ├── modules/            # Coeur ML (Logic de prédiction)
│   ├── models.py           # Modèles Django (Historique)
│   ├── views.py            # Contrôleurs et rendu graphique
│   └── urls.py             # Routes de l'application
├── templates/              # Fichiers HTML (Bootstrap 5)
├── static/                 # CSS, JS et Images
├── manage.py               # Script de gestion Django
└── requirements.txt        # Liste des bibliothèques Python

