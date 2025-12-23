"""
Module pour la classification des plateformes de recrutement
Extrait du notebook Classification.ipynb
Prédit quelle plateforme donnera la meilleure visibilité pour un type de job
"""
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import chi2_contingency
import io
import base64

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class PlatformClassificationModel:
    def __init__(self):
        self.df = None
        self.df_balanced = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_scaled = None
        self.y_encoded = None
        self.label_encoder = None
        self.scaler = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
        # Cache directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, 'model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'classification_models.pkl')
        
    def load_data(self, file_path=None, sample_size=None):
        """Charge les données depuis un fichier CSV"""
        if file_path is None:
            # Chercher le fichier dans le dossier parent
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            file_path = os.path.join(parent_dir, 'jobs_data.csv')
        
        try:
            # Essayer avec séparateur ';'
            if sample_size:
                self.df = pd.read_csv(file_path, encoding='utf-8-sig', sep=';', nrows=sample_size)
            else:
                self.df = pd.read_csv(file_path, encoding='utf-8-sig', sep=';')
        except:
            try:
                if sample_size:
                    self.df = pd.read_csv(file_path, encoding='utf-8', sep=';', nrows=sample_size)
                else:
                    self.df = pd.read_csv(file_path, encoding='utf-8', sep=';')
            except:
                if sample_size:
                    self.df = pd.read_csv(file_path, nrows=sample_size)
                else:
                    self.df = pd.read_csv(file_path)
        
        return self.df
    
    def preprocess_data(self):
        """Prétraite les données"""
        if self.df is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        
        df = self.df.copy()
        
        # Supprimer colonnes inutiles
        cols_to_drop = ['salary_rate', 'salary_year_avg', 'salary_hour_avg', 'job_posted_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Supprimer doublons
        df = df.drop_duplicates()
        
        # Traiter colonnes binaires
        binary_cols = ['job_work_from_home', 'job_no_degree_mention', 'job_health_insurance']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'FAUX': 0, 'VRAI': 1}).fillna(0).astype(int)
        
        # Nettoyer job_via (plateforme)
        if 'job_via' in df.columns:
            df['job_via'] = df['job_via'].fillna('Unknown').replace('', 'Unknown')
            df['job_via'] = df['job_via'].str.replace('via ', '', regex=False).str.strip()
            df['job_via'] = df['job_via'].apply(self._unify_platform)
        
        # Remplir valeurs manquantes
        for col in ['job_location', 'job_country', 'company_name', 'job_title', 'search_location']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Simplifier job_title
        if 'job_title' in df.columns:
            df['job_title_simplified'] = df['job_title'].apply(self._simplify_job_title)
        
        # Filtrer top 10 plateformes
        if 'job_via' in df.columns:
            top_platforms = df['job_via'].value_counts().nlargest(10).index.tolist()
            df = df[df['job_via'].isin(top_platforms)].copy()
        
        # Équilibrage basique des classes (réduit pour performance)
        max_samples = 5000  # Réduit de 15000 à 5000 pour plus de rapidité
        balanced_data = []
        for platform in df['job_via'].unique():
            platform_data = df[df['job_via'] == platform]
            if len(platform_data) > max_samples:
                platform_data = platform_data.sample(n=max_samples, random_state=42)
            balanced_data.append(platform_data)
        
        self.df_balanced = pd.concat(balanced_data, axis=0).sample(frac=1, random_state=42)
        
        return self.df_balanced
    
    def _unify_platform(self, x):
        """Unifie les noms de plateformes"""
        x = str(x)
        if 'LinkedIn' in x: return 'LinkedIn'
        elif 'BeBee' in x: return 'BeBee'
        elif 'Jooble' in x: return 'Jooble'
        elif 'Smart' in x and 'Recruiters' in x: return 'SmartRecruiters'
        elif 'Trabajo' in x: return 'Trabajo.org'
        elif 'Indeed' in x: return 'Indeed'
        elif x in ['?????', 'Unknown', '']: return 'Unknown'
        else: return x
    
    def _simplify_job_title(self, title):
        """Simplifie le titre du poste"""
        title_lower = str(title).lower()
        
        # Niveau de séniorité
        if any(w in title_lower for w in ['senior', 'sr.', 'lead', 'principal', 'staff', 'head', 'chief']):
            level = 'Senior'
        elif any(w in title_lower for w in ['junior', 'jr.', 'entry', 'graduate', 'associate']):
            level = 'Junior'
        elif 'intern' in title_lower:
            level = 'Intern'
        else:
            level = 'Mid'
        
        # Type de poste
        if 'data engineer' in title_lower:
            role = 'Data Engineer'
        elif 'data scien' in title_lower:
            role = 'Data Scientist'
        elif 'data analyst' in title_lower or 'data analy' in title_lower:
            role = 'Data Analyst'
        elif 'machine learning' in title_lower or 'ml engineer' in title_lower:
            role = 'ML Engineer'
        elif 'business analyst' in title_lower or 'business intel' in title_lower:
            role = 'Business Analyst'
        elif 'software engineer' in title_lower or 'software dev' in title_lower:
            role = 'Software Engineer'
        else:
            role = 'Other Tech'
        
        return f"{level} {role}"
    
    def prepare_features(self):
        """Prépare les features pour l'entraînement"""
        if self.df_balanced is None:
            raise ValueError("Données non prétraitées. Appelez preprocess_data() d'abord.")
        
        # Colonnes à supprimer
        cols_to_drop = ['job_via', 'job_type_skills', 'job_title_simplified', 
                       'job_title_short', 'job_location', 'search_location', 
                       'company_name', 'job_title']
        cols_to_drop = [col for col in cols_to_drop if col in self.df_balanced.columns]
        
        X = self.df_balanced.drop(columns=cols_to_drop)
        y = self.df_balanced['job_via']
        
        # Encoder les variables catégorielles
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encoder la cible
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)
        
        # Normalisation
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = X.columns.tolist()
        
        return self.X_scaled, self.y_encoded
    
    def split_data(self, test_size=0.2):
        """Divise les données en ensembles d'entraînement et de test"""
        if self.X_scaled is None:
            self.prepare_features()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y_encoded, test_size=test_size, 
            random_state=42, stratify=self.y_encoded
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_models(self):
        """Sauvegarde les modèles dans un fichier pickle"""
        cache_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def load_models(self):
        """Charge les modèles depuis le cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.models = cache_data['models']
                self.scaler = cache_data['scaler']
                self.label_encoder = cache_data['label_encoder']
                self.feature_names = cache_data['feature_names']
                self.best_model_name = cache_data.get('best_model_name')
                if self.best_model_name and self.best_model_name in self.models:
                    self.best_model = self.models[self.best_model_name]
                return True
            except Exception as e:
                print(f"Erreur lors du chargement du cache: {e}")
                return False
        return False
    
    def train_models(self, force_retrain=False):
        """Entraîne plusieurs modèles de classification"""
        # Essayer de charger depuis le cache
        if not force_retrain and self.load_models():
            print("Modèles chargés depuis le cache")
            return self.models
        
        print("Entraînement des modèles...")
        if self.X_train is None:
            self.split_data()
        
        # KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train, self.y_train)
        self.models['KNN'] = knn
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=10, random_state=42)
        dt.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dt
        
        # XGBoost (si disponible)
        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='mlogloss')
            xgb.fit(self.X_train, self.y_train)
            self.models['XGBoost'] = xgb
        
        # SVM (plus lent, donc paramètres basiques)
        svm = SVC(C=10, kernel='rbf', gamma='scale', random_state=42)
        svm.fit(self.X_train, self.y_train)
        self.models['SVM'] = svm
        
        # Sauvegarder les modèles
        self.save_models()
        print("Modèles entraînés et sauvegardés")
        
        return self.models
    
    def evaluate_models(self):
        """Évalue tous les modèles"""
        if not self.models:
            self.train_models()
        
        results = {}
        best_acc = -1
        
        for name, model in self.models.items():
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            acc_train = accuracy_score(self.y_train, y_pred_train)
            acc_test = accuracy_score(self.y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_scaled, self.y_encoded, cv=5, n_jobs=-1)
            
            results[name] = {
                'train_accuracy': round(acc_train, 4),
                'test_accuracy': round(acc_test, 4),
                'cv_score': round(cv_scores.mean(), 4),
                'overfitting': round(acc_train - acc_test, 4)
            }
            
            if acc_test > best_acc:
                best_acc = acc_test
                self.best_model = model
                self.best_model_name = name
        
        return results
    
    def visualize_comparison(self):
        """Visualise la comparaison des modèles"""
        results = self.evaluate_models()
        
        models = list(results.keys())
        test_acc = [results[m]['test_accuracy'] for m in models]
        train_acc = [results[m]['train_accuracy'] for m in models]
        cv_scores = [results[m]['cv_score'] for m in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy comparison
        x = np.arange(len(models))
        width = 0.25
        
        axes[0].bar(x - width, train_acc, width, label='Train', color='skyblue')
        axes[0].bar(x, test_acc, width, label='Test', color='lightcoral')
        axes[0].bar(x + width, cv_scores, width, label='CV', color='lightgreen')
        axes[0].set_xlabel('Modèles', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Comparaison des Accuracies', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Test accuracy only
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = axes[1].barh(models, test_acc, color=colors, edgecolor='black')
        for bar, acc in zip(bars, test_acc):
            axes[1].text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{acc:.2%}', va='center', fontweight='bold')
        axes[1].set_xlabel('Test Accuracy', fontsize=12)
        axes[1].set_title('Test Accuracy par Modèle', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()
        
        # Overfitting
        overfitting = [results[m]['overfitting'] for m in models]
        colors_over = ['green' if o < 0.02 else 'orange' if o < 0.05 else 'red' for o in overfitting]
        axes[2].barh(models, overfitting, color=colors_over, edgecolor='black')
        for i, (model, o) in enumerate(zip(models, overfitting)):
            axes[2].text(o + 0.002, i, f'{o:.3f}', va='center', fontweight='bold')
        axes[2].set_xlabel('Overfitting (Train - Test)', fontsize=12)
        axes[2].set_title('Overfitting par Modèle', fontsize=14, fontweight='bold')
        axes[2].axvline(x=0.02, color='orange', linestyle='--', alpha=0.5, label='Seuil acceptable')
        axes[2].axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='Seuil critique')
        axes[2].legend()
        axes[2].grid(axis='x', alpha=0.3)
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def confusion_matrix_plot(self, model_name=None):
        """Génère la matrice de confusion"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Valeurs absolues
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Nombre'},
                   ax=axes[0])
        axes[0].set_xlabel('Prédictions', fontsize=12)
        axes[0].set_ylabel('Valeurs Réelles', fontsize=12)
        axes[0].set_title(f'Matrice de Confusion - {model_name}\n(Valeurs absolues)', 
                         fontsize=13, fontweight='bold')
        
        # Pourcentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Proportion'},
                   ax=axes[1])
        axes[1].set_xlabel('Prédictions', fontsize=12)
        axes[1].set_ylabel('Valeurs Réelles', fontsize=12)
        axes[1].set_title(f'Matrice de Confusion - {model_name}\n(Pourcentages)', 
                         fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def get_platforms(self):
        """Retourne la liste des plateformes disponibles"""
        if self.label_encoder is None:
            return []
        return self.label_encoder.classes_.tolist()
    
    def predict_platform(self, job_data, model_name=None):
        """Prédit la meilleure plateforme pour un job"""
        if not self.models:
            self.train_models()
        
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        # Créer un DataFrame avec les données
        input_df = pd.DataFrame([job_data])
        
        # Encoder les colonnes catégorielles comme pendant l'entraînement
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                # Pour les colonnes string, on utilise un encodage simple basé sur le hash
                # ou on pourrait garder les encodeurs originaux
                input_df[col] = input_df[col].astype(str).apply(lambda x: abs(hash(x)) % (10 ** 8))
        
        # S'assurer que toutes les colonnes existent avec des valeurs par défaut
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Réorganiser les colonnes pour correspondre à l'ordre d'entraînement
        input_df = input_df[self.feature_names]
        
        # Normaliser avec le même scaler
        input_scaled = self.scaler.transform(input_df)
        
        # Prédire
        prediction = model.predict(input_scaled)[0]
        probas = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        # Décoder
        platform = self.label_encoder.inverse_transform([prediction])[0]
        
        # Probabilités par plateforme
        if probas is not None:
            platform_probas = {
                self.label_encoder.inverse_transform([i])[0]: round(float(prob * 100), 2)
                for i, prob in enumerate(probas)
            }
            # Trier par probabilité décroissante
            platform_probas = dict(sorted(platform_probas.items(), key=lambda x: x[1], reverse=True))
        else:
            platform_probas = {platform: 100.0}
        
        return platform, platform_probas
    
    def visualize_platform_distribution(self):
        """Visualise la distribution des plateformes dans le dataset"""
        if self.df_balanced is None:
            raise ValueError("Données non prétraitées.")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Distribution des plateformes
        platform_counts = self.df_balanced['job_via'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(platform_counts)))
        
        axes[0].barh(platform_counts.index, platform_counts.values, color=colors, edgecolor='black')
        axes[0].set_xlabel('Nombre de Jobs', fontsize=12)
        axes[0].set_title('Distribution des Jobs par Plateforme', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Pourcentages
        platform_pct = (platform_counts / platform_counts.sum() * 100).round(2)
        axes[1].pie(platform_pct, labels=platform_pct.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[1].set_title('Répartition en Pourcentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_feature_importance(self, model_name=None):
        """Visualise l'importance des features pour le modèle"""
        if not self.models:
            self.train_models()
        
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        # Vérifier si le modèle a feature_importances_
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
        bars = ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor='black')
        
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top 15 Features - {model_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importances[indices[i]]:.4f}',
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_classification_report(self, model_name=None):
        """Visualise le rapport de classification sous forme de heatmap"""
        if not self.models:
            self.train_models()
        
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        from sklearn.metrics import classification_report
        report = classification_report(self.y_test, y_pred, 
                                       target_names=self.label_encoder.classes_,
                                       output_dict=True, zero_division=0)
        
        # Extraire precision, recall, f1-score
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        labels = []
        
        for label in self.label_encoder.classes_:
            if label in report:
                data.append([report[label][m] for m in metrics])
                labels.append(label)
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(labels)
        
        # Ajouter les valeurs
        for i in range(len(labels)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(f'Rapport de Classification - {model_name}', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Score')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
