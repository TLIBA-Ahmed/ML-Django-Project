"""
Module d'Analyse des Emplois (DS2)
Régression: prédiction des salaires
Classification: prédiction du profil IA (High AI / Low AI)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import io
import base64
import os
from django.conf import settings


class JobAnalysisModel:
    """Modèle d'analyse des emplois (régression et classification)"""
    
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(settings.BASE_DIR, 'data_prep_with_clusters.csv')
        self.csv_path = csv_path
        self.df = None
        self.df_encoded = None
        self.onehot_encoder = None
        self.scaler = None
        
        # Modèles de régression
        self.lr_model = None
        self.rfr_model = None
        
        # Modèle de classification
        self.rfc_model = None
        
        # Métriques
        self.regression_metrics = {}
        self.classification_metrics = {}
        
    def load_data(self):
        """Charge les données depuis le CSV"""
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Vérifier colonnes nécessaires
            required_cols = ['salary_usd', 'year_experience', 'nb_skills', 
                           'ai_skill_flag', 'job_description_length', 'benefits_score',
                           'company_size', 'seniority', 'ai_profile', 'sector_group', 
                           'company_location']
            
            for col in required_cols:
                if col not in self.df.columns:
                    raise ValueError(f"Colonne manquante: {col}")
            
            # Nettoyer
            self.df = self.df.dropna(subset=['salary_usd'])
            
            return True
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return False
    
    def prepare_features(self):
        """Prépare les features pour la modélisation"""
        try:
            # Reset index
            df_reset = self.df.reset_index(drop=True)
            
            # Colonnes catégorielles à encoder
            cat_cols = ['company_size', 'seniority', 'ai_profile', 
                       'sector_group', 'company_location']
            
            # One-Hot Encoding
            self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', 
                                               handle_unknown='ignore')
            encoded = self.onehot_encoder.fit_transform(df_reset[cat_cols])
            
            # DataFrame encodé
            encoded_df = pd.DataFrame(encoded, 
                                     columns=self.onehot_encoder.get_feature_names_out(cat_cols))
            
            # Colonnes numériques
            cols_to_keep = ['salary_usd', 'year_experience', 'nb_skills', 
                           'ai_skill_flag', 'job_description_length', 'benefits_score']
            
            # Concaténer
            self.df_encoded = pd.concat([df_reset[cols_to_keep], encoded_df], axis=1)
            
            # Normaliser les colonnes numériques
            num_cols = ['year_experience', 'salary_usd', 'job_description_length', 'benefits_score']
            self.scaler = MinMaxScaler()
            self.df_encoded[num_cols] = self.scaler.fit_transform(self.df_encoded[num_cols])
            
            # Supprimer NaN
            self.df_encoded = self.df_encoded.dropna()
            
            return True
        except Exception as e:
            print(f"Erreur lors de la préparation: {e}")
            return False
    
    def train_regression_models(self):
        """Entraîne les modèles de régression pour prédire le salaire"""
        try:
            # Préparer X et y
            y = self.df_encoded['salary_usd']
            X = self.df_encoded.drop(columns=['salary_usd'])
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Linear Regression
            self.lr_model = LinearRegression()
            self.lr_model.fit(X_train, y_train)
            y_pred_lr = self.lr_model.predict(X_test)
            
            # Random Forest Regressor
            self.rfr_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.rfr_model.fit(X_train, y_train)
            y_pred_rfr = self.rfr_model.predict(X_test)
            
            # Métriques
            self.regression_metrics = {
                'linear_regression': {
                    'mse': mean_squared_error(y_test, y_pred_lr),
                    'r2': r2_score(y_test, y_pred_lr),
                    'y_test': y_test.values,
                    'y_pred': y_pred_lr
                },
                'random_forest': {
                    'mse': mean_squared_error(y_test, y_pred_rfr),
                    'r2': r2_score(y_test, y_pred_rfr),
                    'y_test': y_test.values,
                    'y_pred': y_pred_rfr
                },
                'feature_names': X.columns.tolist()
            }
            
            return self.regression_metrics
        except Exception as e:
            print(f"Erreur lors de l'entraînement régression: {e}")
            return None
    
    def train_classification_model(self):
        """Entraîne le modèle de classification pour prédire le profil IA"""
        try:
            # Target: ai_profile_Low AI (1 = Low AI, 0 = High AI)
            y_class = self.df_encoded['ai_profile_Low AI']
            
            # X: retirer la target ET ai_skill_flag (leakage)
            X_class = self.df_encoded.drop(columns=['ai_profile_Low AI', 'ai_skill_flag'], errors='ignore')
            
            # Split stratifié
            X_train, X_test, y_train, y_test = train_test_split(
                X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
            )
            
            # Random Forest Classifier
            self.rfc_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.rfc_model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = self.rfc_model.predict(X_test)
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            self.classification_metrics = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'y_test': y_test.values,
                'y_pred': y_pred,
                'feature_names': X_class.columns.tolist()
            }
            
            return self.classification_metrics
        except Exception as e:
            print(f"Erreur lors de l'entraînement classification: {e}")
            return None
    
    def predict_salary(self, year_experience, nb_skills, job_description_length, 
                      benefits_score, company_size, seniority, ai_profile, 
                      sector_group, company_location, model_type='random_forest'):
        """Prédit le salaire pour de nouvelles données"""
        try:
            # Créer DataFrame
            input_data = pd.DataFrame({
                'year_experience': [year_experience],
                'nb_skills': [nb_skills],
                'ai_skill_flag': [1 if ai_profile == 'High AI' else 0],
                'job_description_length': [job_description_length],
                'benefits_score': [benefits_score],
                'company_size': [company_size],
                'seniority': [seniority],
                'ai_profile': [ai_profile],
                'sector_group': [sector_group],
                'company_location': [company_location]
            })
            
            # Encoder catégorielles
            cat_cols = ['company_size', 'seniority', 'ai_profile', 'sector_group', 'company_location']
            encoded = self.onehot_encoder.transform(input_data[cat_cols])
            encoded_df = pd.DataFrame(encoded, 
                                     columns=self.onehot_encoder.get_feature_names_out(cat_cols))
            
            # Colonnes numériques
            num_data = input_data[['year_experience', 'nb_skills', 'ai_skill_flag', 
                                   'job_description_length', 'benefits_score']].copy()
            
            # Normaliser (simuler la normalisation avec moyennes raisonnables)
            num_data_scaled = num_data.copy()
            num_data_scaled['year_experience'] = year_experience / 15.0  # Normalisation approximative
            num_data_scaled['job_description_length'] = min(job_description_length / 2500.0, 1.0)
            num_data_scaled['benefits_score'] = benefits_score / 10.0
            
            # Concaténer
            X_input = pd.concat([num_data_scaled, encoded_df], axis=1)
            
            # Assurer l'ordre des colonnes
            expected_cols = self.regression_metrics['feature_names']
            for col in expected_cols:
                if col not in X_input.columns:
                    X_input[col] = 0
            X_input = X_input[expected_cols]
            
            # Prédire
            if model_type == 'linear_regression':
                salary_normalized = self.lr_model.predict(X_input)[0]
            else:  # random_forest
                salary_normalized = self.rfr_model.predict(X_input)[0]
            
            # Dénormaliser (approximatif: salaires entre 30k et 200k)
            salary_usd = salary_normalized * 170000 + 30000
            
            return {
                'salary_usd': float(salary_usd),
                'model_used': model_type
            }
        except Exception as e:
            print(f"Erreur lors de la prédiction salaire: {e}")
            return None
    
    def predict_ai_profile(self, salary_usd, year_experience, nb_skills, 
                          job_description_length, benefits_score, 
                          company_size, seniority, sector_group, company_location):
        """Prédit le profil IA (High AI / Low AI)"""
        try:
            # Créer DataFrame
            input_data = pd.DataFrame({
                'salary_usd': [salary_usd],
                'year_experience': [year_experience],
                'nb_skills': [nb_skills],
                'job_description_length': [job_description_length],
                'benefits_score': [benefits_score],
                'company_size': [company_size],
                'seniority': [seniority],
                'ai_profile': ['Low AI'],  # Placeholder
                'sector_group': [sector_group],
                'company_location': [company_location]
            })
            
            # Encoder
            cat_cols = ['company_size', 'seniority', 'ai_profile', 'sector_group', 'company_location']
            encoded = self.onehot_encoder.transform(input_data[cat_cols])
            encoded_df = pd.DataFrame(encoded, 
                                     columns=self.onehot_encoder.get_feature_names_out(cat_cols))
            
            # Normaliser
            num_data = input_data[['salary_usd', 'year_experience', 
                                   'job_description_length', 'benefits_score']].copy()
            num_data['salary_usd'] = min((salary_usd - 30000) / 170000, 1.0)
            num_data['year_experience'] = year_experience / 15.0
            num_data['job_description_length'] = min(job_description_length / 2500.0, 1.0)
            num_data['benefits_score'] = benefits_score / 10.0
            
            # Ajouter nb_skills
            num_data['nb_skills'] = nb_skills
            
            # Concaténer
            X_input = pd.concat([num_data, encoded_df], axis=1)
            
            # Retirer ai_profile_Low AI et ai_skill_flag
            X_input = X_input.drop(columns=['ai_profile_Low AI'], errors='ignore')
            
            # Assurer l'ordre
            expected_cols = self.classification_metrics['feature_names']
            for col in expected_cols:
                if col not in X_input.columns:
                    X_input[col] = 0
            X_input = X_input[expected_cols]
            
            # Prédire
            prediction = self.rfc_model.predict(X_input)[0]
            proba = self.rfc_model.predict_proba(X_input)[0]
            
            ai_profile = 'Low AI' if prediction == 1 else 'High AI'
            confidence = float(proba[prediction])
            
            return {
                'ai_profile': ai_profile,
                'confidence': confidence,
                'probabilities': {'Low AI': float(proba[1]), 'High AI': float(proba[0])}
            }
        except Exception as e:
            print(f"Erreur lors de la prédiction profil IA: {e}")
            return None
    
    def plot_regression_comparison(self):
        """Compare les modèles de régression"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Linear Regression
            lr_metrics = self.regression_metrics['linear_regression']
            ax1.scatter(lr_metrics['y_test'], lr_metrics['y_pred'], alpha=0.3, color='blue')
            max_val = max(lr_metrics['y_test'].max(), lr_metrics['y_pred'].max())
            min_val = min(lr_metrics['y_test'].min(), lr_metrics['y_pred'].min())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Prédiction parfaite')
            ax1.set_title(f'Régression Linéaire\nR² = {lr_metrics["r2"]:.4f}')
            ax1.set_xlabel('Salaire Réel (Normalisé)')
            ax1.set_ylabel('Salaire Prédit (Normalisé)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Random Forest
            rfr_metrics = self.regression_metrics['random_forest']
            ax2.scatter(rfr_metrics['y_test'], rfr_metrics['y_pred'], alpha=0.3, color='green')
            max_val = max(rfr_metrics['y_test'].max(), rfr_metrics['y_pred'].max())
            min_val = min(rfr_metrics['y_test'].min(), rfr_metrics['y_pred'].min())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Prédiction parfaite')
            ax2.set_title(f'Random Forest Regressor\nR² = {rfr_metrics["r2"]:.4f}')
            ax2.set_xlabel('Salaire Réel (Normalisé)')
            ax2.set_ylabel('Salaire Prédit (Normalisé)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'data:image/png;base64,{image_base64}'
        except Exception as e:
            print(f"Erreur visualisation régression: {e}")
            return None
    
    def plot_feature_importance(self):
        """Affiche l'importance des features pour Random Forest Regressor"""
        try:
            importances = self.rfr_model.feature_importances_
            feature_names = self.regression_metrics['feature_names']
            
            # Trier
            indices = np.argsort(importances)[::-1][:15]
            
            df_importance = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Importance', y='Feature', data=df_importance, palette='viridis')
            plt.title(f'Top 15 Features Importantes (Random Forest Regressor)\nR² = {self.regression_metrics["random_forest"]["r2"]:.4f}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'data:image/png;base64,{image_base64}'
        except Exception as e:
            print(f"Erreur visualisation importance: {e}")
            return None
    
    def plot_confusion_matrix(self):
        """Affiche la matrice de confusion pour la classification"""
        try:
            cm = self.classification_metrics['confusion_matrix']
            accuracy = self.classification_metrics['accuracy']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['High AI', 'Low AI'],
                       yticklabels=['High AI', 'Low AI'])
            plt.title(f'Matrice de Confusion (Random Forest Classifier)\nAccuracy = {accuracy:.4f}')
            plt.xlabel('Valeurs Prédites')
            plt.ylabel('Valeurs Réelles')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'data:image/png;base64,{image_base64}'
        except Exception as e:
            print(f"Erreur visualisation confusion matrix: {e}")
            return None
    
    def plot_classification_report(self):
        """Visualise le rapport de classification"""
        try:
            report = self.classification_metrics['classification_report']
            
            # Extraire métriques
            classes = ['High AI (0)', 'Low AI (1)']
            metrics = ['precision', 'recall', 'f1-score']
            
            data = {
                'Class': [],
                'Metric': [],
                'Score': []
            }
            
            for i, class_label in enumerate(['0', '1']):
                if class_label in report:
                    for metric in metrics:
                        data['Class'].append(classes[i])
                        data['Metric'].append(metric.replace('-', ' ').title())
                        data['Score'].append(report[class_label][metric])
            
            df_report = pd.DataFrame(data)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_report, x='Class', y='Score', hue='Metric', palette='Set2')
            plt.title(f'Rapport de Classification\nAccuracy = {self.classification_metrics["accuracy"]:.4f}')
            plt.ylim(0, 1)
            plt.ylabel('Score')
            plt.legend(title='Metric')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'data:image/png;base64,{image_base64}'
        except Exception as e:
            print(f"Erreur visualisation rapport: {e}")
            return None
