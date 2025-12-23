"""
Module pour l'analyse de clustering des jobs AI
Extrait du notebook Clustering.ipynb
"""
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour Django
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
import io
import base64
import kagglehub


class ClusteringAnalysis:
    def __init__(self):
        self.df = None
        self.df_encoded = None
        self.X_scaled = None
        self.scaler = None
        self.label_encoders = {}
        self.kmeans_model = None
        self.pca_model = None
        self.X_pca = None
        
        # Cache directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, 'model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'clustering_models.pkl')
        
    def load_data(self):
        """Charge les données depuis le fichier local"""
        # Chercher le fichier dans le répertoire parent du projet Django
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.dirname(base_dir)
        csv_path = os.path.join(parent_dir, 'ai_job_dataset.csv')
        
        if not os.path.exists(csv_path):
            # Fallback: télécharger depuis Kaggle si le fichier local n'existe pas
            path = kagglehub.dataset_download("bismasajjad/global-ai-job-market-and-salary-trends-2025")
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            csv_path = os.path.join(path, csv_files[0])
        
        self.df = pd.read_csv(csv_path)
        return self.df
    
    def preprocess_data(self):
        """Prétraite les données pour le clustering"""
        if self.df is None:
            self.load_data()
        
        self.df_encoded = self.df.copy()
        
        # Encoder les variables catégorielles
        for col in self.df_encoded.columns:
            if self.df_encoded[col].dtype == 'object' or isinstance(self.df_encoded[col].dtype, pd.CategoricalDtype):
                le = LabelEncoder()
                self.df_encoded[col] = le.fit_transform(self.df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        # Gérer les valeurs manquantes
        self.df_encoded = self.df_encoded.fillna(self.df_encoded.median(numeric_only=True))
        
        # Normalisation
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.df_encoded)
        
        return self.X_scaled
    
    def perform_pca(self, n_components=2):
        """Effectue une ACP"""
        if self.X_scaled is None:
            self.preprocess_data()
        
        self.pca_model = PCA(n_components=n_components)
        self.X_pca = self.pca_model.fit_transform(self.X_scaled)
        
        return self.X_pca
    
    def elbow_method(self, k_range=range(2, 11)):
        """Méthode du coude pour déterminer le nombre optimal de clusters"""
        if self.X_pca is None:
            self.perform_pca()
        
        inertia = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_pca)
            inertia.append(kmeans.inertia_)
        
        # Créer le graphique
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertia, marker='o', linewidth=2)
        plt.xlabel('Nombre de clusters (k)', fontsize=12)
        plt.ylabel('Inertie', fontsize=12)
        plt.title('Méthode du coude - K-means', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def save_models(self):
        """Sauvegarde les modèles dans un fichier pickle"""
        cache_data = {
            'kmeans_model': self.kmeans_model,
            'pca_model': self.pca_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def load_models(self):
        """Charge les modèles depuis le cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.kmeans_model = cache_data['kmeans_model']
                self.pca_model = cache_data['pca_model']
                self.scaler = cache_data['scaler']
                self.label_encoders = cache_data['label_encoders']
                return True
            except Exception as e:
                print(f"Erreur lors du chargement du cache: {e}")
                return False
        return False
    
    def perform_kmeans(self, n_clusters=4, force_retrain=False):
        """Effectue le clustering K-means"""
        # Essayer de charger depuis le cache
        if not force_retrain and self.load_models():
            print("Modèles de clustering chargés depuis le cache")
            # Recréer X_pca si nécessaire
            if self.X_pca is None and self.pca_model is not None and self.X_scaled is not None:
                self.X_pca = self.pca_model.transform(self.X_scaled)
        else:
            print("Entraînement du modèle de clustering...")
            if self.X_pca is None:
                self.perform_pca()
            
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = self.kmeans_model.fit_predict(self.X_pca)
            
            # Sauvegarder les modèles
            self.save_models()
            print("Modèles de clustering sauvegardés")
        
        # Prédire les clusters
        clusters = self.kmeans_model.predict(self.X_pca)
        
        # Ajouter les clusters au dataframe
        self.df_encoded['cluster'] = clusters
        self.df_encoded['PC1'] = self.X_pca[:, 0]
        self.df_encoded['PC2'] = self.X_pca[:, 1]
        
        return clusters
    
    def visualize_clusters(self):
        """Visualise les clusters sur les composantes PCA"""
        if self.kmeans_model is None:
            self.perform_kmeans()
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.X_pca[:, 0],
            self.X_pca[:, 1],
            c=self.df_encoded['cluster'],
            cmap='viridis',
            s=50,
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
        
        # Ajouter les centres des clusters
        centers = self.kmeans_model.cluster_centers_
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            c='red',
            s=300,
            marker='X',
            edgecolors='black',
            linewidth=2,
            label='Centres des clusters'
        )
        
        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.title('Visualisation des clusters (K-means sur PCA)', fontsize=14)
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def cluster_profiles(self):
        """Génère les profils de clusters"""
        if 'cluster' not in self.df_encoded.columns:
            self.perform_kmeans()
        
        # Récupérer les colonnes numériques d'origine
        numeric_cols = []
        for col in ['salary_usd', 'years_experience', 'remote_ratio', 'benefits_score']:
            if col in self.df.columns:
                numeric_cols.append(col)
        
        profiles = {}
        
        for cluster_id in sorted(self.df_encoded['cluster'].unique()):
            cluster_data = self.df_encoded[self.df_encoded['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(self.df_encoded) * 100, 2)
            }
            
            # Statistiques numériques
            for col in numeric_cols:
                if col in self.df.columns:
                    original_values = self.df.loc[cluster_data.index, col]
                    profile[col] = {
                        'mean': round(original_values.mean(), 2),
                        'median': round(original_values.median(), 2),
                        'min': round(original_values.min(), 2),
                        'max': round(original_values.max(), 2)
                    }
            
            # Catégories dominantes
            for col in ['experience_level', 'company_location', 'company_size']:
                if col in self.df.columns:
                    dominant = self.df.loc[cluster_data.index, col].mode()
                    if len(dominant) > 0:
                        profile[f'{col}_dominant'] = dominant[0]
            
            profiles[f'Cluster {cluster_id}'] = profile
        
        return profiles
    
    def predict_cluster(self, job_data):
        """Prédit le cluster pour de nouvelles données de job"""
        if self.kmeans_model is None:
            self.perform_kmeans()
        
        # Créer un DataFrame avec les mêmes colonnes
        input_df = pd.DataFrame([job_data])
        
        # Encoder les variables catégorielles
        for col in input_df.columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                try:
                    input_df[col] = le.transform([input_df[col].iloc[0]])
                except:
                    # Si la valeur n'existe pas, utiliser la valeur la plus fréquente
                    input_df[col] = 0
        
        # S'assurer que toutes les colonnes existent
        for col in self.df_encoded.columns:
            if col not in input_df.columns and col not in ['cluster', 'PC1', 'PC2']:
                input_df[col] = 0
        
        # Réorganiser les colonnes
        input_df = input_df[[col for col in self.df_encoded.columns if col not in ['cluster', 'PC1', 'PC2']]]
        
        # Normaliser
        input_scaled = self.scaler.transform(input_df)
        
        # PCA
        input_pca = self.pca_model.transform(input_scaled)
        
        # Prédire
        cluster = self.kmeans_model.predict(input_pca)[0]
        
        return int(cluster)
    
    def visualize_cluster_distributions(self):
        """Visualise la distribution des features numériques par cluster"""
        if 'cluster' not in self.df_encoded.columns:
            self.perform_kmeans()
        
        numeric_features = ['salary_usd', 'years_experience', 'remote_ratio', 'benefits_score']
        available_features = [f for f in numeric_features if f in self.df.columns]
        
        if not available_features:
            return None
        
        n_features = len(available_features)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features))
        
        if n_features == 1:
            axes = [axes]
        
        for idx, feature in enumerate(available_features):
            for cluster_id in sorted(self.df_encoded['cluster'].unique()):
                cluster_data = self.df.loc[self.df_encoded['cluster'] == cluster_id, feature]
                axes[idx].hist(cluster_data, bins=30, alpha=0.6, label=f'Cluster {cluster_id}')
            
            axes[idx].set_xlabel(feature, fontsize=12)
            axes[idx].set_ylabel('Fréquence', fontsize=12)
            axes[idx].set_title(f'Distribution de {feature} par Cluster', fontsize=14, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_cluster_sizes(self):
        """Visualise la taille des clusters"""
        if 'cluster' not in self.df_encoded.columns:
            self.perform_kmeans()
        
        cluster_sizes = self.df_encoded['cluster'].value_counts().sort_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Diagramme en barres
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
        axes[0].bar(cluster_sizes.index, cluster_sizes.values, color=colors, edgecolor='black', width=0.6)
        axes[0].set_xlabel('Cluster', fontsize=12)
        axes[0].set_ylabel('Nombre de Jobs', fontsize=12)
        axes[0].set_title('Taille des Clusters', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, (cluster, size) in enumerate(cluster_sizes.items()):
            axes[0].text(cluster, size + 50, str(size), ha='center', fontweight='bold')
        
        # Diagramme circulaire
        axes[1].pie(cluster_sizes.values, labels=[f'Cluster {i}' for i in cluster_sizes.index],
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1].set_title('Répartition en Pourcentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_cluster_profiles_comparison(self):
        """Visualise la comparaison des profils de clusters"""
        if 'cluster' not in self.df_encoded.columns:
            self.perform_kmeans()
        
        numeric_features = ['salary_usd', 'years_experience', 'remote_ratio', 'benefits_score']
        available_features = [f for f in numeric_features if f in self.df.columns]
        
        if not available_features:
            return None
        
        # Calculer les moyennes par cluster
        cluster_means = self.df.groupby(self.df_encoded['cluster'])[available_features].mean()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(available_features))
        width = 0.2
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_means)))
        
        for i, (cluster_id, row) in enumerate(cluster_means.iterrows()):
            offset = (i - len(cluster_means)/2) * width
            bars = ax.bar(x + offset, row.values, width, label=f'Cluster {cluster_id}', 
                         color=colors[i], edgecolor='black')
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Valeur Moyenne', fontsize=12)
        ax.set_title('Comparaison des Moyennes par Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(available_features, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
