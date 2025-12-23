"""
Module pour la prédiction de salaire
Extrait du notebook DS1.ipynb
"""
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour Django
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io
import base64
import kagglehub


class SalaryPredictionModel:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        # Cache directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, 'model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'salary_models.pkl')
        
    def load_data(self):
        """Charge les données depuis le fichier local"""
        # Chercher le fichier dans le répertoire parent du projet Django
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.dirname(base_dir)
        csv_path = os.path.join(parent_dir, 'Salary_Data.csv')
        
        if not os.path.exists(csv_path):
            # Fallback: télécharger depuis Kaggle si le fichier local n'existe pas
            path = kagglehub.dataset_download("ayeshasajjad123/salary-data")
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            csv_path = os.path.join(path, csv_files[0])
        
        self.df = pd.read_csv(csv_path)
        return self.df
    
    def preprocess_data(self):
        """Prétraite les données"""
        if self.df is None:
            self.load_data()
        
        # Gérer les valeurs manquantes
        for col in ['Age', 'Years of Experience', 'Salary']:
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        for col in ['Gender', 'Education Level', 'Job Title']:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        # Encoder les variables catégorielles
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.df[col]):
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
        
        return self.df
    
    def split_data(self, test_size=0.2):
        """Divise les données en ensembles d'entraînement et de test"""
        if self.df is None:
            self.preprocess_data()
        
        X = self.df.drop('Salary', axis=1)
        y = self.df['Salary']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_models(self):
        """Sauvegarde les modèles dans un fichier pickle"""
        cache_data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
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
                self.label_encoders = cache_data['label_encoders']
                self.best_model_name = cache_data.get('best_model_name')
                if self.best_model_name and self.best_model_name in self.models:
                    model_data = self.models[self.best_model_name]
                    if isinstance(model_data, dict):
                        self.best_model = model_data['model']
                    else:
                        self.best_model = model_data
                return True
            except Exception as e:
                print(f"Erreur lors du chargement du cache: {e}")
                return False
        return False
    
    def train_models(self, force_retrain=False):
        """Entraîne plusieurs modèles de régression"""
        # Essayer de charger depuis le cache
        if not force_retrain and self.load_models():
            print("Modèles chargés depuis le cache")
            return self.models
        
        print("Entraînement des modèles...")
        if self.X_train is None:
            self.split_data()
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr_model
        
        # Decision Tree
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_model.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dt_model
        
        # Random Forest
        rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        
        # Gradient Boosting
        gbr_model = GradientBoostingRegressor(random_state=42)
        gbr_model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gbr_model
        
        # Polynomial Regression
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train)
        X_test_poly = poly.transform(self.X_test)
        
        poly_lr_model = LinearRegression()
        poly_lr_model.fit(X_train_poly, self.y_train)
        self.models['Polynomial Regression'] = {
            'model': poly_lr_model,
            'poly': poly,
            'X_test_poly': X_test_poly
        }
        
        # Sauvegarder les modèles
        self.save_models()
        print("Modèles entraînés et sauvegardés")
        
        return self.models
    
    def evaluate_models(self):
        """Évalue tous les modèles"""
        if not self.models:
            self.train_models()
        
        results = {}
        best_r2 = -float('inf')
        
        for name, model in self.models.items():
            if name == 'Polynomial Regression':
                y_pred = model['model'].predict(model['X_test_poly'])
            else:
                y_pred = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'MAE': round(mae, 2),
                'MSE': round(mse, 2),
                'RMSE': round(np.sqrt(mse), 2),
                'R2': round(r2, 4)
            }
            
            # Garder le meilleur modèle
            if r2 > best_r2:
                best_r2 = r2
                self.best_model_name = name
                self.best_model = model
        
        return results
    
    def visualize_comparison(self):
        """Visualise la comparaison des modèles"""
        results = self.evaluate_models()
        
        # Créer un DataFrame pour les métriques
        metrics_df = pd.DataFrame(results).T
        
        # Créer 3 graphiques
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # MAE
        metrics_df['MAE'].plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('MAE', fontsize=12)
        axes[0].set_xlabel('Modèle', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # RMSE
        metrics_df['RMSE'].plot(kind='bar', ax=axes[1], color='lightcoral', edgecolor='black')
        axes[1].set_title('Root Mean Squared Error (RMSE)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_xlabel('Modèle', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        # R2 Score
        metrics_df['R2'].plot(kind='bar', ax=axes[2], color='lightgreen', edgecolor='black')
        axes[2].set_title('R² Score', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('R² Score', fontsize=12)
        axes[2].set_xlabel('Modèle', fontsize=12)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_predictions(self, model_name=None):
        """Visualise les prédictions vs valeurs réelles pour un modèle"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if model_name == 'Polynomial Regression':
            y_pred = model['model'].predict(model['X_test_poly'])
        else:
            y_pred = model.predict(self.X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', linewidth=2, label='Prédiction parfaite')
        plt.xlabel('Salaire Réel', fontsize=12)
        plt.ylabel('Salaire Prédit', fontsize=12)
        plt.title(f'{model_name}: Prédictions vs Valeurs Réelles', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ajouter R² sur le graphique
        r2 = r2_score(self.y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def correlation_matrix(self):
        """Génère la matrice de corrélation"""
        if self.df is None:
            self.preprocess_data()
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                   linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
        plt.title('Matrice de Corrélation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def predict_salary(self, input_data, model_name=None):
        """Prédit le salaire pour de nouvelles données"""
        if not self.models:
            self.train_models()
        
        if model_name is None:
            model_name = self.best_model_name
        
        # Créer un DataFrame avec les données d'entrée
        input_df = pd.DataFrame([input_data])
        
        # Encoder les variables catégorielles
        for col in input_df.columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                try:
                    input_df[col] = le.transform([input_df[col].iloc[0]])
                except:
                    # Si la valeur n'existe pas, utiliser 0
                    input_df[col] = 0
        
        # Faire la prédiction
        model = self.models[model_name]
        
        if model_name == 'Polynomial Regression':
            poly = model['poly']
            input_poly = poly.transform(input_df)
            prediction = model['model'].predict(input_poly)[0]
        else:
            prediction = model.predict(input_df)[0]
        
        return round(float(prediction), 2)
    
    def visualize_salary_distribution(self):
        """Visualise la distribution des salaires dans le dataset"""
        if self.df is None:
            self.preprocess_data()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogramme
        axes[0].hist(self.df['Salary'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Salaire', fontsize=12)
        axes[0].set_ylabel('Fréquence', fontsize=12)
        axes[0].set_title('Distribution des Salaires', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axvline(self.df['Salary'].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: ${self.df["Salary"].mean():.0f}')
        axes[0].axvline(self.df['Salary'].median(), color='green', linestyle='--', linewidth=2, label=f'Médiane: ${self.df["Salary"].median():.0f}')
        axes[0].legend()
        
        # Boxplot
        axes[1].boxplot(self.df['Salary'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       capprops=dict(color='black', linewidth=1.5))
        axes[1].set_ylabel('Salaire', fontsize=12)
        axes[1].set_title('Boxplot des Salaires', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_salary_by_category(self):
        """Visualise les salaires par catégories"""
        if self.df is None:
            self.preprocess_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Par niveau d'éducation
        if 'Education Level' in self.df.columns:
            education_salary = self.df.groupby('Education Level')['Salary'].mean().sort_values()
            axes[0, 0].barh(education_salary.index, education_salary.values, color='lightcoral', edgecolor='black')
            axes[0, 0].set_xlabel('Salaire Moyen', fontsize=12)
            axes[0, 0].set_title('Salaire Moyen par Niveau d\'Éducation', fontsize=14, fontweight='bold')
            axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Par titre de poste
        if 'Job Title' in self.df.columns:
            job_salary = self.df.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(10)
            axes[0, 1].barh(job_salary.index, job_salary.values, color='lightgreen', edgecolor='black')
            axes[0, 1].set_xlabel('Salaire Moyen', fontsize=12)
            axes[0, 1].set_title('Top 10 Postes par Salaire Moyen', fontsize=14, fontweight='bold')
            axes[0, 1].grid(axis='x', alpha=0.3)
            axes[0, 1].invert_yaxis()
        
        # Par genre
        if 'Gender' in self.df.columns:
            gender_salary = self.df.groupby('Gender')['Salary'].mean()
            colors = ['lightblue' if g == 'Male' else 'pink' for g in gender_salary.index]
            axes[1, 0].bar(gender_salary.index, gender_salary.values, color=colors, edgecolor='black')
            axes[1, 0].set_ylabel('Salaire Moyen', fontsize=12)
            axes[1, 0].set_title('Salaire Moyen par Genre', fontsize=14, fontweight='bold')
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for i, v in enumerate(gender_salary.values):
                axes[1, 0].text(i, v + 1000, f'${v:.0f}', ha='center', fontweight='bold')
        
        # Salaire vs Années d'expérience
        if 'Years of Experience' in self.df.columns:
            axes[1, 1].scatter(self.df['Years of Experience'], self.df['Salary'], alpha=0.5, s=30, edgecolors='w', linewidth=0.5)
            axes[1, 1].set_xlabel('Années d\'Expérience', fontsize=12)
            axes[1, 1].set_ylabel('Salaire', fontsize=12)
            axes[1, 1].set_title('Salaire vs Années d\'Expérience', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Ligne de tendance
            z = np.polyfit(self.df['Years of Experience'], self.df['Salary'], 1)
            p = np.poly1d(z)
            axes[1, 1].plot(self.df['Years of Experience'].sort_values(), 
                          p(self.df['Years of Experience'].sort_values()), 
                          "r--", linewidth=2, label='Tendance')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_residuals(self, model_name=None):
        """Visualise les résidus du modèle"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if model_name == 'Polynomial Regression':
            y_pred = model['model'].predict(model['X_test_poly'])
        else:
            y_pred = model.predict(self.X_test)
        
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Résidus vs Prédictions
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Valeurs Prédites', fontsize=12)
        axes[0].set_ylabel('Résidus', fontsize=12)
        axes[0].set_title(f'{model_name}: Résidus vs Prédictions', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution des résidus
        axes[1].hist(residuals, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Résidus', fontsize=12)
        axes[1].set_ylabel('Fréquence', fontsize=12)
        axes[1].set_title(f'{model_name}: Distribution des Résidus', fontsize=14, fontweight='bold')
        axes[1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {residuals.mean():.2f}')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
