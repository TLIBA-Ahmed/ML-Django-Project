"""
Module pour la prédiction de séries temporelles
Extrait du notebook DSobjectif2.ipynb
Analyse et prévision du nombre d'offres d'emploi par mois
"""
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io
import base64
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesForecastModel:
    def __init__(self):
        self.df = None
        self.serie_mensuelle = None
        self.decomposition = None
        self.saisonnalite = None
        self.best_model_name = "SMA(6) + STL"
        
        # Cache directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, 'model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'timeseries_models.pkl')
        
    def load_data(self):
        """Charge les données depuis le fichier local"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.dirname(base_dir)
        csv_path = os.path.join(parent_dir, 'dataset_final3.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset non trouvé: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # Convertir posting_date en datetime
        self.df['posting_date'] = pd.to_datetime(self.df['posting_date'])
        
        return self.df
    
    def prepare_time_series(self):
        """Prépare la série temporelle mensuelle"""
        if self.df is None:
            self.load_data()
        
        # Créer une colonne Mois_Annee
        self.df['Mois_Annee'] = self.df['posting_date'].dt.to_period('M')
        
        # Compter les offres par mois
        self.serie_mensuelle = self.df.groupby('Mois_Annee').size()
        
        # Convertir l'index en datetime
        self.serie_mensuelle.index = self.serie_mensuelle.index.to_timestamp()
        
        # Définir la fréquence mensuelle
        self.serie_mensuelle = self.serie_mensuelle.asfreq('MS')
        
        # Donner un nom
        self.serie_mensuelle.name = 'Nombre_Offres'
        
        return self.serie_mensuelle
    
    def decompose_series(self):
        """Décompose la série temporelle avec STL"""
        if self.serie_mensuelle is None:
            self.prepare_time_series()
        
        # Décomposition STL
        self.decomposition = STL(self.serie_mensuelle, seasonal=13).fit()
        self.saisonnalite = self.decomposition.seasonal
        
        return self.decomposition
    
    def save_models(self):
        """Sauvegarde les modèles dans un fichier pickle"""
        cache_data = {
            'serie_mensuelle': self.serie_mensuelle,
            'saisonnalite': self.saisonnalite,
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
                self.serie_mensuelle = cache_data['serie_mensuelle']
                self.saisonnalite = cache_data['saisonnalite']
                self.best_model_name = cache_data.get('best_model_name')
                return True
            except Exception as e:
                print(f"Erreur lors du chargement du cache: {e}")
                return False
        return False
    
    def get_statistics(self):
        """Retourne les statistiques de la série temporelle"""
        if self.serie_mensuelle is None:
            self.prepare_time_series()
        
        stats = {
            'total': int(self.serie_mensuelle.sum()),
            'moyenne': round(self.serie_mensuelle.mean(), 0),
            'ecart_type': round(self.serie_mensuelle.std(), 0),
            'minimum': int(self.serie_mensuelle.min()),
            'mois_min': self.serie_mensuelle.idxmin().strftime('%B %Y'),
            'maximum': int(self.serie_mensuelle.max()),
            'mois_max': self.serie_mensuelle.idxmax().strftime('%B %Y')
        }
        
        return stats
    
    def visualize_time_series(self):
        """Visualise la série temporelle"""
        if self.serie_mensuelle is None:
            self.prepare_time_series()
        
        # Créer le tableau avec les noms de mois
        if 'Mois' in self.df.columns and 'Lib_Mois' in self.df.columns:
            tableau_mois = self.df.groupby(['Mois', 'Lib_Mois']).size().reset_index(name='Nombre_Offres')
            tableau_mois = tableau_mois.sort_values('Mois')
            noms_mois = tableau_mois['Lib_Mois'].tolist()
            valeurs = tableau_mois['Nombre_Offres'].tolist()
        else:
            noms_mois = [date.strftime('%B') for date in self.serie_mensuelle.index]
            valeurs = self.serie_mensuelle.values.tolist()
        
        plt.figure(figsize=(12, 6))
        
        # Tracer la courbe
        plt.plot(noms_mois, valeurs, marker='o', linewidth=2, markersize=10, 
                color='steelblue', label='Nombre d\'offres')
        
        # Ligne de moyenne
        moyenne = np.mean(valeurs)
        plt.axhline(y=moyenne, color='red', linestyle='--', linewidth=2, 
                   label=f'Moyenne: {moyenne:.0f}')
        
        # Annotations pour le pic et le creux
        idx_max = np.argmax(valeurs)
        idx_min = np.argmin(valeurs)
        plt.annotate(f'Pic: {valeurs[idx_max]}', xy=(idx_max, valeurs[idx_max]), 
                    xytext=(idx_max, valeurs[idx_max] + 30),
                    ha='center', fontsize=11, color='green', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        plt.annotate(f'Creux: {valeurs[idx_min]}', xy=(idx_min, valeurs[idx_min]), 
                    xytext=(idx_min, valeurs[idx_min] - 30),
                    ha='center', fontsize=11, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        plt.title('Série Temporelle: Nombre d\'Offres par Mois', fontsize=14, fontweight='bold')
        plt.xlabel('Mois', fontsize=12)
        plt.ylabel('Nombre d\'Offres', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_decomposition(self):
        """Visualise la décomposition STL"""
        if self.decomposition is None:
            self.decompose_series()
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        # 1. Série originale
        axes[0].plot(self.serie_mensuelle.index, self.serie_mensuelle.values, 
                    color='steelblue', linewidth=2)
        axes[0].set_title('Série Temporelle Originale', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Nombre d\'Offres', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Tendance
        axes[1].plot(self.decomposition.trend.index, self.decomposition.trend.values, 
                    color='orange', linewidth=2)
        axes[1].set_title('Tendance (Trend)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Tendance', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # 3. Saisonnalité
        axes[2].plot(self.decomposition.seasonal.index, self.decomposition.seasonal.values, 
                    color='green', linewidth=2)
        axes[2].set_title('Saisonnalité (Seasonality)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Saisonnalité', fontsize=11)
        axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[2].grid(True, alpha=0.3)
        
        # 4. Résidus
        axes[3].plot(self.decomposition.resid.index, self.decomposition.resid.values, 
                    color='red', linewidth=1)
        axes[3].set_title('Résidus (Residuals)', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Résidus', fontsize=11)
        axes[3].set_xlabel('Date', fontsize=11)
        axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def compare_models(self):
        """Compare différents modèles de prédiction"""
        if self.serie_mensuelle is None:
            self.prepare_time_series()
        
        if self.saisonnalite is None:
            self.decompose_series()
        
        # Split train/test (9 premiers mois / 3 derniers mois)
        train = self.serie_mensuelle[:9]
        test = self.serie_mensuelle[9:]
        
        results = {}
        
        # 1. Modèle Holt (sans saisonnalité)
        try:
            model_holt = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
            pred_holt = model_holt.forecast(steps=3)
            mape_holt = np.mean(np.abs((test.values - pred_holt.values) / test.values)) * 100
            results['Holt'] = round(mape_holt, 1)
        except:
            results['Holt'] = None
        
        # 2. Holt + STL
        if results['Holt'] is not None:
            saisonnalite_test = [self.saisonnalite.iloc[9], self.saisonnalite.iloc[10], self.saisonnalite.iloc[11]]
            pred_holt_stl = pred_holt.values + saisonnalite_test
            mape_holt_stl = np.mean(np.abs((test.values - pred_holt_stl) / test.values)) * 100
            results['Holt + STL'] = round(mape_holt_stl, 1)
        
        # 3. SMA(3) + STL
        sma_3 = train[-3:].mean()
        pred_sma3_stl = [sma_3 + s for s in saisonnalite_test]
        mape_sma3_stl = np.mean(np.abs((test.values - pred_sma3_stl) / test.values)) * 100
        results['SMA(3) + STL'] = round(mape_sma3_stl, 1)
        
        # 4. SMA(6) + STL (meilleur modèle)
        sma_6 = train[-6:].mean()
        pred_sma6_stl = [sma_6 + s for s in saisonnalite_test]
        mape_sma6_stl = np.mean(np.abs((test.values - pred_sma6_stl) / test.values)) * 100
        results['SMA(6) + STL'] = round(mape_sma6_stl, 1)
        
        return results
    
    def visualize_model_comparison(self):
        """Visualise la comparaison des modèles"""
        results = self.compare_models()
        
        # Filtrer les None
        results = {k: v for k, v in results.items() if v is not None}
        
        plt.figure(figsize=(10, 6))
        
        models = list(results.keys())
        mapes = list(results.values())
        colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
        
        bars = plt.bar(models, mapes, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
        
        # Ajouter les valeurs sur les barres
        for bar, mape in zip(bars, mapes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mape:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.title('Comparaison des Modèles (MAPE)', fontsize=14, fontweight='bold')
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.xlabel('Modèle', fontsize=12)
        plt.xticks(rotation=15, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def predict_next_months(self, num_months=3):
        """Prédit les prochains mois avec le meilleur modèle (SMA(6) + STL)"""
        if self.serie_mensuelle is None:
            self.prepare_time_series()
        
        if self.saisonnalite is None:
            self.decompose_series()
        
        # Calculer SMA sur les 6 derniers mois
        sma_6 = self.serie_mensuelle[-6:].mean()
        
        # Générer les dates futures
        last_date = self.serie_mensuelle.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                     periods=num_months, freq='MS')
        
        # Prédictions avec saisonnalité
        predictions = []
        for i in range(num_months):
            # Utiliser la saisonnalité du mois correspondant (0-11)
            month_idx = (last_date.month + i) % 12
            saisonnalite_val = self.saisonnalite.iloc[month_idx]
            pred = sma_6 + saisonnalite_val
            predictions.append(round(pred, 0))
        
        result = {
            'dates': [d.strftime('%B %Y') for d in future_dates],
            'predictions': predictions,
            'sma_6': round(sma_6, 0),
            'model': self.best_model_name
        }
        
        return result
    
    def visualize_predictions(self, num_months=3):
        """Visualise les prédictions futures"""
        predictions_data = self.predict_next_months(num_months)
        
        # Générer les dates futures
        last_date = self.serie_mensuelle.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                     periods=num_months, freq='MS')
        
        plt.figure(figsize=(14, 7))
        
        # Données réelles
        plt.plot(self.serie_mensuelle.index, self.serie_mensuelle.values, 
                marker='o', linewidth=2, color='steelblue', 
                label='Données réelles 2023')
        
        # Prédictions
        plt.plot(future_dates, predictions_data['predictions'], 
                marker='s', linewidth=3, linestyle='--', color='green', 
                markersize=10, label=f'Prédictions 2024 ({self.best_model_name})')
        
        # Annotations
        for date, pred in zip(future_dates, predictions_data['predictions']):
            plt.annotate(f'{int(pred)}', xy=(date, pred), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=11, fontweight='bold', color='green')
        
        plt.title('Prédictions du Nombre d\'Offres pour 2024', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Nombre d\'Offres', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
