"""
Module pour la prédiction des années d'expérience basé sur les caractéristiques du poste
Utilise plusieurs modèles de régression: Linear Regression, Random Forest, Gradient Boosting, SVR
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import io
import base64
import os


class ExperiencePredictionModel:
    """Classe pour gérer la prédiction des années d'expérience"""
    
    def __init__(self, dataset_path=None):
        """
        Initialise le modèle de prédiction d'expérience
        
        Args:
            dataset_path: Chemin vers le dataset CSV
        """
        if dataset_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dataset_path = os.path.join(base_dir, 'Full_dataset.csv')
        
        self.dataset_path = dataset_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.encoders = {}
        self.feature_columns = ['industry', 'education_required', 'required_skills', 
                                'employment_type', 'experience_level', 'company_size']
        self.target_column = 'years_experience'
        
        # Charger et préparer les données
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Charge et prépare les données pour l'entraînement"""
        # Charger le dataset
        self.df = pd.read_csv(self.dataset_path, sep=';', engine='python')
        
        # Encoder les features catégorielles
        df_encoded = self.df.copy()
        for col in self.feature_columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le
        
        # Préparer X et y
        X = df_encoded[self.feature_columns]
        y = df_encoded[self.target_column]
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
    def train_all_models(self):
        """Entraîne tous les modèles de régression"""
        # Linear Regression
        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(self.X_train, self.y_train)
        
        # Random Forest
        self.models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        
        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        )
        self.models['Gradient Boosting'].fit(self.X_train, self.y_train)
        
        # SVR with StandardScaler
        self.models['SVR'] = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='auto'))
        self.models['SVR'].fit(self.X_train, self.y_train)
        
    def evaluate_models(self):
        """Évalue tous les modèles et stocke les métriques"""
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            self.predictions[name] = y_pred
            
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            self.metrics[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
    def get_best_model(self):
        """Retourne le meilleur modèle basé sur le R²"""
        if not self.metrics:
            self.evaluate_models()
        
        best_model_name = max(self.metrics, key=lambda x: self.metrics[x]['R2'])
        return best_model_name, self.metrics[best_model_name]
    
    def visualize_model_comparison(self):
        """Crée un graphique comparant tous les modèles"""
        if not self.metrics:
            self.evaluate_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparaison des Modèles de Régression', fontsize=16, fontweight='bold')
        
        models = list(self.metrics.keys())
        
        # MAE
        maes = [self.metrics[m]['MAE'] for m in models]
        axes[0, 0].bar(models, maes, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE
        rmses = [self.metrics[m]['RMSE'] for m in models]
        axes[0, 1].bar(models, rmses, color='lightcoral', edgecolor='darkred')
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R²
        r2s = [self.metrics[m]['R2'] for m in models]
        axes[1, 0].bar(models, r2s, color='lightgreen', edgecolor='darkgreen')
        axes[1, 0].set_title('R² Score', fontweight='bold')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        # Metrics table
        axes[1, 1].axis('off')
        table_data = []
        for m in models:
            table_data.append([
                m,
                f"{self.metrics[m]['MAE']:.3f}",
                f"{self.metrics[m]['RMSE']:.3f}",
                f"{self.metrics[m]['R2']:.3f}"
            ])
        
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Model', 'MAE', 'RMSE', 'R²'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def visualize_actual_vs_predicted(self, model_name):
        """Crée un graphique Actual vs Predicted pour un modèle spécifique"""
        if model_name not in self.predictions:
            self.evaluate_models()
        
        y_pred = self.predictions[model_name]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5, edgecolor='k')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Années d\'Expérience Réelles', fontsize=12)
        plt.ylabel('Années d\'Expérience Prédites', fontsize=12)
        plt.title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def visualize_residuals(self, model_name):
        """Crée un graphique des résidus pour un modèle"""
        if model_name not in self.predictions:
            self.evaluate_models()
        
        y_pred = self.predictions[model_name]
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{model_name}: Analyse des Résidus', fontsize=14, fontweight='bold')
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolor='k')
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Valeurs Prédites', fontsize=11)
        axes[0].set_ylabel('Résidus', fontsize=11)
        axes[0].set_title('Résidus vs Prédictions')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[1].set_xlabel('Résidus', fontsize=11)
        axes[1].set_ylabel('Fréquence', fontsize=11)
        axes[1].set_title('Distribution des Résidus')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def predict_experience(self, input_data):
        """
        Prédit les années d'expérience pour de nouvelles données
        
        Args:
            input_data: dict avec les features {industry, education_required, ...}
        
        Returns:
            dict avec les prédictions de tous les modèles
        """
        # Encoder les inputs
        encoded_input = {}
        for col in self.feature_columns:
            if col in input_data:
                try:
                    encoded_input[col] = self.encoders[col].transform([input_data[col]])[0]
                except:
                    # Si la valeur n'existe pas dans l'encoder, utiliser la première valeur connue
                    encoded_input[col] = 0
        
        # Créer un DataFrame pour la prédiction
        X_input = pd.DataFrame([encoded_input], columns=self.feature_columns)
        
        # Prédire avec tous les modèles
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_input)[0]
            predictions[name] = round(pred, 2)
        
        # Trouver le meilleur modèle
        best_model_name, _ = self.get_best_model()
        
        return {
            'predictions': predictions,
            'best_model': best_model_name,
            'best_prediction': predictions[best_model_name]
        }
    
    def get_feature_importance(self):
        """Retourne l'importance des features pour Random Forest"""
        if 'Random Forest' not in self.models:
            return None
        
        rf_model = self.models['Random Forest']
        importances = rf_model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def visualize_feature_importance(self):
        """Visualise l'importance des features"""
        feature_importance = self.get_feature_importance()
        
        if feature_importance is None:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'], color='teal', edgecolor='black')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Importance des Features (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
