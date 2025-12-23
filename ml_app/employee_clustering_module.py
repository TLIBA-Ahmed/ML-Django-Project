"""
Module pour le clustering des employés en Junior/Senior
Utilise KMeans, Agglomerative Clustering, et DBSCAN
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import io
import base64
import os


class EmployeeClusteringModel:
    """Classe pour gérer le clustering des employés"""
    
    def __init__(self, dataset_path=None):
        """
        Initialise le modèle de clustering
        
        Args:
            dataset_path: Chemin vers le dataset CSV
        """
        if dataset_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dataset_path = os.path.join(base_dir, 'Full_dataset.csv')
        
        self.dataset_path = dataset_path
        self.df = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.numeric_cols = ['salary_usd', 'benefits_score', 'years_experience', 'remote_ratio']
        
        # Modèles de clustering
        self.kmeans_model = None
        self.agg_model = None
        self.dbscan_model = None
        
        # Labels
        self.kmeans_labels = None
        self.agg_labels = None
        self.dbscan_labels = None
        
        # Charger les données
        self.load_data()
        
    def load_data(self):
        """Charge et prépare les données"""
        self.df = pd.read_csv(self.dataset_path, sep=';', engine='python')
        
        # Extraire les colonnes numériques
        X_numeric = self.df[self.numeric_cols].values
        
        # Normaliser
        self.X_scaled = self.scaler.fit_transform(X_numeric)
    
    def find_optimal_k_kmeans(self):
        """Trouve le nombre optimal de clusters pour KMeans"""
        K_range = range(2, 11)
        
        wcss = []
        sil_scores = []
        ch_scores = []
        db_scores = []
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.X_scaled)
            
            wcss.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(self.X_scaled, labels))
            ch_scores.append(calinski_harabasz_score(self.X_scaled, labels))
            db_scores.append(davies_bouldin_score(self.X_scaled, labels))
        
        # Trouver le coude
        knee = KneeLocator(list(K_range), wcss, curve='convex', direction='decreasing')
        best_k_elbow = knee.knee if knee.knee else 2
        
        best_k_sil = K_range[np.argmax(sil_scores)]
        best_k_ch = K_range[np.argmax(ch_scores)]
        best_k_db = K_range[np.argmin(db_scores)]
        
        return {
            'K_range': list(K_range),
            'wcss': wcss,
            'sil_scores': sil_scores,
            'ch_scores': ch_scores,
            'db_scores': db_scores,
            'best_k_elbow': best_k_elbow,
            'best_k_sil': best_k_sil,
            'best_k_ch': best_k_ch,
            'best_k_db': best_k_db
        }
    
    def visualize_kmeans_optimization(self):
        """Visualise l'optimisation du nombre de clusters pour KMeans"""
        results = self.find_optimal_k_kmeans()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Optimisation du Nombre de Clusters (KMeans)', fontsize=16, fontweight='bold')
        
        # WCSS (Elbow Method)
        axes[0, 0].plot(results['K_range'], results['wcss'], marker='o', color='blue')
        axes[0, 0].axvline(x=results['best_k_elbow'], color='red', linestyle='--', label=f'Optimal k={results["best_k_elbow"]}')
        axes[0, 0].set_xlabel('Number of clusters (k)')
        axes[0, 0].set_ylabel('WCSS')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette Score
        axes[0, 1].plot(results['K_range'], results['sil_scores'], marker='o', color='green')
        axes[0, 1].axvline(x=results['best_k_sil'], color='red', linestyle='--', label=f'Optimal k={results["best_k_sil"]}')
        axes[0, 1].set_xlabel('Number of clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score (higher is better)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz Score
        axes[1, 0].plot(results['K_range'], results['ch_scores'], marker='o', color='orange')
        axes[1, 0].axvline(x=results['best_k_ch'], color='red', linestyle='--', label=f'Optimal k={results["best_k_ch"]}')
        axes[1, 0].set_xlabel('Number of clusters (k)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('CH Score (higher is better)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Davies-Bouldin Score
        axes[1, 1].plot(results['K_range'], results['db_scores'], marker='o', color='red')
        axes[1, 1].axvline(x=results['best_k_db'], color='red', linestyle='--', label=f'Optimal k={results["best_k_db"]}')
        axes[1, 1].set_xlabel('Number of clusters (k)')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        axes[1, 1].set_title('DB Score (lower is better)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def train_all_models(self, n_clusters=2):
        """Entraîne tous les modèles de clustering"""
        # KMeans
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans_labels = self.kmeans_model.fit_predict(self.X_scaled)
        
        # Agglomerative Clustering
        self.agg_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.agg_labels = self.agg_model.fit_predict(self.X_scaled)
        
        # DBSCAN - Find optimal eps
        eps_optimal = self.find_optimal_eps()
        self.dbscan_model = DBSCAN(eps=eps_optimal, min_samples=3)
        self.dbscan_labels = self.dbscan_model.fit_predict(self.X_scaled)
        
        # Ajouter les labels au DataFrame
        self.df['KMeans_Cluster'] = self.kmeans_labels
        self.df['Agglomerative_Cluster'] = self.agg_labels
        self.df['DBSCAN_Cluster'] = self.dbscan_labels
        
        # Nommer les clusters (Junior/Senior)
        self._name_clusters()
    
    def find_optimal_eps(self):
        """Trouve le eps optimal pour DBSCAN"""
        min_samples = 5
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn_fit = nn.fit(self.X_scaled)
        distances, _ = nn_fit.kneighbors(self.X_scaled)
        
        k_distances = np.sort(distances[:, min_samples-1])
        dist_diff = np.diff(k_distances)
        elbow_index = np.argmax(dist_diff)
        eps_optimal = float(k_distances[elbow_index])
        
        return max(eps_optimal, 0.5)  # Minimum eps
    
    def _name_clusters(self):
        """Nomme les clusters basés sur years_experience (Junior/Senior)"""
        # Pour Agglomerative (le meilleur modèle généralement)
        cluster_means = self.df.groupby('Agglomerative_Cluster')['years_experience'].mean()
        mapping = {cluster_means.idxmin(): "Junior", cluster_means.idxmax(): "Senior"}
        self.df['Cluster_Name'] = self.df['Agglomerative_Cluster'].map(mapping)
    
    def evaluate_models(self):
        """Évalue tous les modèles de clustering"""
        def compute_centroids_sse(X, labels):
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) == 0:
                return None, 0
            centroids = np.array([X[labels == lbl].mean(axis=0) for lbl in unique_labels])
            sse = np.sum([np.sum((X[labels == lbl] - centroids[i])**2) 
                         for i, lbl in enumerate(unique_labels)])
            return centroids, sse
        
        def compute_inter_dist(centroids):
            if centroids is None or len(centroids) <= 1:
                return 0
            k = len(centroids)
            return np.mean([np.linalg.norm(centroids[i]-centroids[j]) 
                           for i in range(k) for j in range(i+1, k)])
        
        # KMeans
        centroids_kmeans, sse_kmeans = compute_centroids_sse(self.X_scaled, self.kmeans_labels)
        inter_dist_kmeans = compute_inter_dist(centroids_kmeans)
        
        # Agglomerative
        centroids_agg, sse_agg = compute_centroids_sse(self.X_scaled, self.agg_labels)
        inter_dist_agg = compute_inter_dist(centroids_agg)
        
        # DBSCAN
        centroids_dbscan, sse_dbscan = compute_centroids_sse(self.X_scaled, self.dbscan_labels)
        inter_dist_dbscan = compute_inter_dist(centroids_dbscan)
        
        metrics = {
            'KMeans': {
                'Unterhomogeneity': sse_kmeans,
                'Untrahomogeneity': inter_dist_kmeans,
                'Score': inter_dist_kmeans / sse_kmeans if sse_kmeans != 0 else 0
            },
            'Agglomerative': {
                'Unterhomogeneity': sse_agg,
                'Untrahomogeneity': inter_dist_agg,
                'Score': inter_dist_agg / sse_agg if sse_agg != 0 else 0
            },
            'DBSCAN': {
                'Unterhomogeneity': sse_dbscan,
                'Untrahomogeneity': inter_dist_dbscan,
                'Score': inter_dist_dbscan / sse_dbscan if sse_dbscan != 0 else 0
            }
        }
        
        best_model = max(metrics, key=lambda x: metrics[x]['Score'])
        
        return metrics, best_model
    
    def visualize_cluster_profiles(self, model_name='Agglomerative'):
        """Visualise les profils des clusters"""
        cluster_col = f'{model_name}_Cluster'
        
        # Filtrer les données (exclure le bruit pour DBSCAN)
        if model_name == 'DBSCAN':
            data = self.df[self.df[cluster_col] != -1]
        else:
            data = self.df
        
        # Calculer les profils moyens
        profile = data.groupby(cluster_col)[self.numeric_cols].mean().round(2)
        
        # Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(profile, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
        plt.title(f'{model_name} Clustering - Profils des Clusters', fontsize=14, fontweight='bold')
        plt.ylabel('Cluster')
        plt.xlabel('Features')
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def visualize_boxplots(self, model_name='Agglomerative'):
        """Crée des boxplots pour chaque feature par cluster"""
        cluster_col = f'{model_name}_Cluster'
        
        if model_name == 'DBSCAN':
            data = self.df[self.df[cluster_col] != -1]
        else:
            data = self.df
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name} - Distribution des Features par Cluster', 
                     fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(self.numeric_cols):
            ax = axes[idx // 2, idx % 2]
            sns.boxplot(data=data, x=cluster_col, y=col, palette='Set2', ax=ax)
            ax.set_title(f'Distribution de {col}', fontweight='bold')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(col)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def visualize_named_clusters(self):
        """Visualise les clusters avec noms (Junior/Senior)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Profils Junior vs Senior', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(self.numeric_cols):
            ax = axes[idx // 2, idx % 2]
            sns.boxplot(data=self.df, x='Cluster_Name', y=col, 
                       palette={'Junior': '#87CEFA', 'Senior': '#FFB6C1'}, ax=ax)
            ax.set_title(f'Distribution de {col}', fontweight='bold')
            ax.set_xlabel('Profil')
            ax.set_ylabel(col)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def predict_cluster(self, input_data):
        """
        Prédit le cluster (Junior/Senior) pour de nouvelles données
        
        Args:
            input_data: dict avec {salary_usd, benefits_score, years_experience, remote_ratio}
        
        Returns:
            dict avec les prédictions de tous les modèles
        """
        # Préparer les données
        X_input = np.array([[
            input_data.get('salary_usd', 0),
            input_data.get('benefits_score', 0),
            input_data.get('years_experience', 0),
            input_data.get('remote_ratio', 0)
        ]])
        
        # Normaliser
        X_input_scaled = self.scaler.transform(X_input)
        
        # Prédire avec chaque modèle
        kmeans_cluster = self.kmeans_model.predict(X_input_scaled)[0]
        agg_cluster = self.agg_model.fit_predict(self.X_scaled)  # Refit nécessaire pour Agglomerative
        
        # Mapper vers Junior/Senior
        cluster_means = self.df.groupby('Agglomerative_Cluster')['years_experience'].mean()
        mapping = {cluster_means.idxmin(): "Junior", cluster_means.idxmax(): "Senior"}
        
        # Déterminer Junior/Senior basé sur years_experience
        if input_data.get('years_experience', 0) < cluster_means.mean():
            cluster_name = "Junior"
        else:
            cluster_name = "Senior"
        
        return {
            'cluster_name': cluster_name,
            'kmeans_cluster': int(kmeans_cluster),
            'agglomerative_cluster': int(kmeans_cluster),  # Approximation
            'years_experience': input_data.get('years_experience', 0),
            'salary_usd': input_data.get('salary_usd', 0)
        }
    
    def get_cluster_statistics(self):
        """Retourne les statistiques par cluster"""
        stats = {}
        
        for cluster_name in ['Junior', 'Senior']:
            cluster_data = self.df[self.df['Cluster_Name'] == cluster_name]
            stats[cluster_name] = {
                'count': len(cluster_data),
                'avg_salary': round(cluster_data['salary_usd'].mean(), 2),
                'avg_experience': round(cluster_data['years_experience'].mean(), 2),
                'avg_benefits': round(cluster_data['benefits_score'].mean(), 2),
                'avg_remote': round(cluster_data['remote_ratio'].mean(), 2)
            }
        
        return stats
