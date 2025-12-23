"""
Module pour la classification des types de postes (Job Title Classification)
Extrait du notebook DS12.ipynb
Prédit le type de poste (Data Analyst, Data Scientist, Data Engineer, etc.)
basé sur les caractéristiques du job
"""
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            f1_score, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
import io
import base64

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class JobTitleClassificationModel:
    """
    Modèle de classification pour prédire le type de poste (job_title_short)
    basé sur les features encodées du dataset
    """
    
    def __init__(self):
        self.df = None
        self.df_ml = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_res = None  # Données après SMOTE
        self.y_train_res = None  # Cible après SMOTE
        self.label_encoders = {}
        self.target_encoder = None
        self.le_target = None  # LabelEncoder pour la cible
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.features_to_use = [
            'job_schedule_type_enc',
            'sector_enc',
            'job_via_enc',
            'job_skills'
        ]
        
        # Cache directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, 'model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'job_title_classification_models.pkl')
    
    def load_data(self, file_path=None):
        """Charge les données depuis un fichier CSV"""
        if file_path is None:
            # Chercher le fichier dans le dossier "integration ML"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # ml_app -> ml_django_project -> integration ML
            integration_ml_dir = os.path.dirname(os.path.dirname(current_dir))
            file_path = os.path.join(integration_ml_dir, 'dataset_final3.csv')
            
            # Si toujours pas trouvé, essayer le dossier parent
            if not os.path.exists(file_path):
                parent_dir = os.path.dirname(integration_ml_dir)
                file_path = os.path.join(parent_dir, 'dataset_final3.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier {file_path} introuvable. Veuillez placer dataset_final3.csv dans le dossier 'integration ML'")
        
        self.df = pd.read_csv(file_path)
        return self.df
    
    def preprocess_data(self):
        """Prétraite les données selon la logique du notebook DS12.ipynb"""
        if self.df is None:
            self.load_data()
        
        # Copie de travail
        self.df_ml = self.df.copy()
        
        # Supprimer les doublons
        self.df_ml = self.df_ml.drop_duplicates()
        
        # Traiter les valeurs manquantes pour sector et sector_group
        self.df_ml['sector'] = self.df_ml['sector'].fillna("Unknown")
        self.df_ml['sector_group'] = self.df_ml['sector_group'].fillna("Unknown")
        
        # Encodage de la variable cible
        self.le_target = LabelEncoder()
        self.df_ml['job_title_short_enc'] = self.le_target.fit_transform(self.df_ml['job_title_short'])
        
        # Colonnes catégorielles à encoder
        cat_cols = [
            'company',
            'job_location',
            'job_country',
            'job_via',
            'job_schedule_type',
            'sector_group',
            'sector'
        ]
        
        # Encodage des colonnes catégorielles avec LabelEncoder
        for col in cat_cols:
            if col in self.df_ml.columns:
                le = LabelEncoder()
                self.df_ml[col + '_enc'] = le.fit_transform(self.df_ml[col])
                self.label_encoders[col] = le
        
        # Target Encoding pour job_title (optionnel, mais améliore les résultats)
        if 'job_title' in self.df_ml.columns:
            y = self.df_ml['job_title_short_enc']
            self.target_encoder = TargetEncoder(cols=['job_title'])
            self.df_ml['job_title_te'] = self.target_encoder.fit_transform(self.df_ml['job_title'], y)
        
        # Supprimer les colonnes problématiques identifiées dans l'analyse
        cols_to_remove = [
            'job_title_te',  # Corrélation parfaite avec la cible (data leakage)
            'company_enc',   # Cardinalité trop élevée
            'job_location_enc',  # 100% de leakage
            'job_country_enc',   # 100% de leakage
        ]
        
        cols_to_remove_existing = [col for col in cols_to_remove if col in self.df_ml.columns]
        if cols_to_remove_existing:
            self.df_ml = self.df_ml.drop(columns=cols_to_remove_existing)
        
        return self.df_ml
    
    def split_data(self, test_size=0.2):
        """Divise les données en ensembles d'entraînement et de test"""
        if self.df_ml is None:
            self.preprocess_data()
        
        # Sélection des features et de la cible
        X = self.df_ml[self.features_to_use]
        y = self.df_ml['job_title_short_enc']
        
        # Division stratifiée pour conserver la distribution des classes
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def apply_smote(self):
        """Applique SMOTE pour équilibrer les classes"""
        if self.X_train is None:
            self.split_data()
        
        # Application de SMOTE
        sm = SMOTE(random_state=42)
        self.X_train_res, self.y_train_res = sm.fit_resample(self.X_train, self.y_train)
        
        return self.X_train_res, self.y_train_res
    
    def save_models(self):
        """Sauvegarde les modèles dans un fichier pickle"""
        cache_data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'le_target': self.le_target,
            'target_encoder': self.target_encoder,
            'best_model_name': self.best_model_name,
            'features_to_use': self.features_to_use
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
                self.label_encoders = cache_data['label_encoders']
                self.le_target = cache_data['le_target']
                self.target_encoder = cache_data.get('target_encoder')
                self.best_model_name = cache_data.get('best_model_name')
                self.features_to_use = cache_data.get('features_to_use', self.features_to_use)
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
        
        # Appliquer SMOTE si pas encore fait
        if self.X_train_res is None:
            self.apply_smote()
        
        # KNN
        print("Entraînement KNN...")
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(self.X_train_res, self.y_train_res)
        self.models['KNN'] = knn
        
        # SVM (avec kernel linéaire pour plus de rapidité)
        print("Entraînement SVM...")
        svm = SVC(kernel='linear', random_state=42, probability=True)
        svm.fit(self.X_train_res, self.y_train_res)
        self.models['SVM'] = svm
        
        # Decision Tree
        print("Entraînement Decision Tree...")
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(self.X_train_res, self.y_train_res)
        self.models['Decision Tree'] = dt
        
        # Random Forest
        print("Entraînement Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )
        rf.fit(self.X_train_res, self.y_train_res)
        self.models['Random Forest'] = rf
        
        # XGBoost (si disponible)
        if XGBOOST_AVAILABLE:
            print("Entraînement XGBoost...")
            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
            xgb.fit(self.X_train_res, self.y_train_res)
            self.models['XGBoost'] = xgb
        
        # Sauvegarder les modèles
        self.save_models()
        print("Modèles entraînés et sauvegardés")
        
        return self.models
    
    def evaluate_models(self):
        """Évalue tous les modèles"""
        if not self.models:
            self.train_models()
        
        results = {}
        best_f1 = -float('inf')
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            results[name] = {
                'Accuracy': round(accuracy, 4),
                'F1_Score': round(f1, 4)
            }
            
            # Garder le meilleur modèle (basé sur F1-score)
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name
                self.best_model = model
        
        return results
    
    def visualize_comparison(self):
        """Visualise la comparaison des modèles"""
        results = self.evaluate_models()
        
        # Créer un DataFrame pour les métriques
        metrics_df = pd.DataFrame(results).T
        
        # Créer 2 graphiques
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        metrics_df['Accuracy'].plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title('Accuracy par modèle', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_xlabel('Modèle', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.0])
        
        # F1_Score
        metrics_df['F1_Score'].plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
        axes[1].set_title('F1-Score par modèle', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('F1-Score', fontsize=12)
        axes[1].set_xlabel('Modèle', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_confusion_matrix(self, model_name=None):
        """Visualise la matrice de confusion pour un modèle"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        # Matrice de confusion
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classes names
        class_names = self.le_target.classes_
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={"shrink": 0.8})
        plt.title(f'Matrice de Confusion - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Classe Réelle', fontsize=12)
        plt.xlabel('Classe Prédite', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_class_distribution(self):
        """Visualise la distribution des classes avant et après SMOTE"""
        if self.y_train is None:
            self.split_data()
        if self.y_train_res is None:
            self.apply_smote()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Distribution avant SMOTE
        class_counts_before = pd.Series(self.y_train).value_counts().sort_index()
        class_names = self.le_target.classes_
        
        axes[0].bar(range(len(class_counts_before)), class_counts_before.values, 
                   color='lightcoral', edgecolor='black')
        axes[0].set_xticks(range(len(class_names)))
        axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0].set_title('Distribution des classes AVANT SMOTE', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Nombre d\'exemples', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Distribution après SMOTE
        class_counts_after = pd.Series(self.y_train_res).value_counts().sort_index()
        
        axes[1].bar(range(len(class_counts_after)), class_counts_after.values, 
                   color='lightgreen', edgecolor='black')
        axes[1].set_xticks(range(len(class_names)))
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].set_title('Distribution des classes APRÈS SMOTE', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Nombre d\'exemples', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_knn_k_values(self):
        """Visualise F1-score pour différentes valeurs de k (KNN)"""
        if self.X_train_res is None:
            self.apply_smote()
        
        f1_scores = []
        k_values = range(1, 31)
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train_res, self.y_train_res)
            y_pred = knn.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            f1_scores.append(f1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(k_values, f1_scores, marker='o', linestyle='dashed', color='green', linewidth=2)
        plt.title('F1-score en fonction du nombre de voisins (k)', fontsize=14, fontweight='bold')
        plt.xlabel('Nombre de voisins (k)', fontsize=12)
        plt.ylabel('F1-score (weighted)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Marquer le meilleur k
        best_k = k_values[f1_scores.index(max(f1_scores))]
        best_f1 = max(f1_scores)
        plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, 
                   label=f'Meilleur k = {best_k} (F1 = {best_f1:.4f})')
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_roc_curves(self):
        """Visualise les courbes ROC pour tous les modèles"""
        if not self.models:
            self.train_models()
        
        # Binariser y_test pour ROC multiclasse
        classes = np.unique(self.y_test)
        y_test_bin = label_binarize(self.y_test, classes=classes)
        n_classes = len(classes)
        
        plt.figure(figsize=(12, 8))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
        
        for (model_name, model), color in zip(self.models.items(), colors):
            # Obtenir les probabilités
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_test)
            else:
                # Pour SVM sans probability=True
                continue
            
            # Calculer ROC curve moyenne
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Macro-average ROC curve
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            
            mean_auc = auc(all_fpr, mean_tpr)
            
            plt.plot(all_fpr, mean_tpr, color=color, linewidth=2,
                    label=f'{model_name} (AUC = {mean_auc:.4f})')
        
        # Ligne diagonale (modèle aléatoire)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Aléatoire (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
        plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
        plt.title('Courbes ROC - Comparaison des modèles', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_decision_tree(self):
        """Visualise l'arbre de décision"""
        if 'Decision Tree' not in self.models:
            self.train_models()
        
        dt = self.models['Decision Tree']
        class_names = self.le_target.classes_
        
        plt.figure(figsize=(25, 15))
        plot_tree(
            dt, 
            feature_names=self.features_to_use,
            class_names=[str(c) for c in class_names],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title('Visualisation de l\'Arbre de Décision', fontsize=16, fontweight='bold')
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def get_classification_report(self, model_name=None):
        """Obtient le rapport de classification détaillé"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        class_names = self.le_target.classes_
        report = classification_report(self.y_test, y_pred, 
                                      target_names=class_names,
                                      output_dict=True)
        
        return report
    
    def predict_job_title(self, input_data, model_name=None):
        """
        Prédit le type de poste pour de nouvelles données
        
        Args:
            input_data (dict): Dictionnaire avec les features nécessaires
                Exemple: {
                    'job_schedule_type': 'Full-time',
                    'sector': 'Information Technology',
                    'job_via': 'LinkedIn',
                    'job_skills': 3
                }
            model_name (str): Nom du modèle à utiliser (optionnel)
        
        Returns:
            str: Type de poste prédit
        """
        if not self.models:
            self.train_models()
        
        if model_name is None:
            model_name = self.best_model_name
        
        # Créer un DataFrame avec les données d'entrée
        input_df = pd.DataFrame([input_data])
        
        # Encoder les variables catégorielles
        for col in ['job_schedule_type', 'sector', 'job_via']:
            col_enc = col + '_enc'
            if col in self.label_encoders and col in input_df.columns:
                le = self.label_encoders[col]
                try:
                    input_df[col_enc] = le.transform([input_df[col].iloc[0]])
                except:
                    # Si la valeur n'existe pas, utiliser la valeur la plus fréquente
                    input_df[col_enc] = 0
                input_df = input_df.drop(columns=[col])
        
        # S'assurer que toutes les features sont présentes
        for feature in self.features_to_use:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Réorganiser les colonnes
        input_df = input_df[self.features_to_use]
        
        # Faire la prédiction
        model = self.models[model_name]
        prediction_enc = model.predict(input_df)[0]
        
        # Décoder la prédiction
        prediction = self.le_target.inverse_transform([prediction_enc])[0]
        
        # Obtenir la probabilité si disponible
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            confidence = np.max(proba)
            return prediction, confidence
        
        return prediction, None
