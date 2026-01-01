"""
Module pour le clustering des jobs basé sur les compétences (skills)
Extrait du notebook Clustering.ipynb
Analyse les profils de compétences et crée des groupes de jobs similaires
"""
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour Django
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from math import pi
import io
import base64


class JobSkillClusteringModel:
    """Modèle de clustering basé sur les compétences des jobs"""
    
    def __init__(self):
        self.df = None  # DataFrame avec les compétences binaires
        self.data = None  # DataFrame complet avec job_title, etc.
        self.df_scaled = None
        self.scaler = None
        self.kmeans_model = None
        self.pca_model = None
        self.X_pca = None
        self.optimal_k = 3  # Basé sur l'analyse du notebook
        
        # Cluster labels basés sur l'analyse du notebook
        self.cluster_label_map = {
            0: "Business / BI Analysts",
            1: "Senior Data Engineers (Big Data & Cloud)",
            2: "Data Scientists & Applied Analysts"
        }
        
        # Cache directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, 'model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'job_skill_clustering_models.pkl')
    
    def load_data(self):
        """Charge les données de compétences depuis processed_data_jobs.csv"""
        # Chercher le fichier dans le répertoire parent du projet Django
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.dirname(base_dir)
        
        # Fichier de compétences prétraité
        processed_path = os.path.join(parent_dir, 'processed_data_jobs.csv')
        # Fichier de données complètes
        df_final_path = os.path.join(parent_dir, 'df_final_ml.csv')
        
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Le fichier processed_data_jobs.csv n'existe pas dans {parent_dir}")
        
        # Charger les compétences binaires
        self.df = pd.read_csv(processed_path)
        
        # Charger les données complètes si disponibles
        if os.path.exists(df_final_path):
            self.data = pd.read_csv(df_final_path)
        else:
            self.data = self.df.copy()
        
        return self.df, self.data
    
    def preprocess_data(self):
        """Prétraite les données pour le clustering"""
        if self.df is None:
            self.load_data()
        
        # Standardiser les features (compétences binaires)
        self.scaler = StandardScaler()
        self.df_scaled = self.scaler.fit_transform(self.df)
        
        return self.df_scaled
    
    def perform_pca(self, n_components=2):
        """Effectue une réduction de dimensionnalité avec PCA"""
        if self.df_scaled is None:
            self.preprocess_data()
        
        self.pca_model = PCA(n_components=n_components, random_state=42)
        self.X_pca = self.pca_model.fit_transform(self.df_scaled)
        
        variance_explained = self.pca_model.explained_variance_ratio_.sum()
        print(f"Variance expliquée par {n_components} composantes: {variance_explained:.3f}")
        
        return self.X_pca, variance_explained
    
    def elbow_method(self, k_range=range(2, 11)):
        """Méthode du coude pour déterminer le nombre optimal de clusters"""
        if self.df_scaled is None:
            self.preprocess_data()
        
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.df)
            inertias.append(kmeans.inertia_)
        
        # Créer le graphique
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Nombre de clusters (K)', fontsize=12)
        plt.ylabel('Inertie', fontsize=12)
        plt.title('Méthode du Coude - Clustering basé sur les Compétences', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64, inertias
    
    def silhouette_analysis(self, k_range=range(2, 11), sample_size=1000):
        """Analyse du score de silhouette"""
        if self.df is None:
            self.load_data()
        
        # Échantillonner les données pour accélérer le calcul
        df_sample = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(df_sample)
            score = silhouette_score(df_sample, labels)
            silhouette_scores.append(score)
        
        # Créer le graphique
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='green')
        plt.xlabel('Nombre de clusters (K)', fontsize=12)
        plt.ylabel('Score de Silhouette', fontsize=12)
        plt.title('Score de Silhouette vs K (Données Échantillonnées)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Convertir en base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64, silhouette_scores
    
    def save_models(self):
        """Sauvegarde les modèles dans un fichier pickle"""
        cache_data = {
            'kmeans_model': self.kmeans_model,
            'pca_model': self.pca_model,
            'scaler': self.scaler,
            'optimal_k': self.optimal_k
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Modèles sauvegardés dans le cache")
    
    def load_models(self):
        """Charge les modèles depuis le cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.kmeans_model = cache_data['kmeans_model']
                self.pca_model = cache_data['pca_model']
                self.scaler = cache_data['scaler']
                self.optimal_k = cache_data['optimal_k']
                print("Modèles chargés depuis le cache")
                return True
            except Exception as e:
                print(f"Erreur lors du chargement du cache: {e}")
                return False
        return False
    
    def perform_kmeans(self, n_clusters=None, force_retrain=False):
        """Effectue le clustering K-means"""
        if n_clusters is None:
            n_clusters = self.optimal_k
        
        # Essayer de charger depuis le cache
        if not force_retrain and self.load_models():
            if self.df_scaled is None:
                self.preprocess_data()
        else:
            print(f"Entraînement du modèle K-means avec k={n_clusters}...")
            if self.df_scaled is None:
                self.preprocess_data()
            
            # Entraîner K-means sur les données de compétences
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
            clusters = self.kmeans_model.fit_predict(self.df)
            
            # Calculer les métriques
            inertia = self.kmeans_model.inertia_
            db_score = davies_bouldin_score(self.df, clusters)
            ch_score = calinski_harabasz_score(self.df, clusters)
            
            print(f"✓ Clustering terminé!")
            print(f"  - Inertie: {inertia:.2f}")
            print(f"  - Davies-Bouldin Score: {db_score:.4f}")
            print(f"  - Calinski-Harabasz Score: {ch_score:.2f}")
            
            # Sauvegarder les modèles
            self.save_models()
        
        # Prédire les clusters
        clusters = self.kmeans_model.predict(self.df)
        
        # Ajouter les clusters aux dataframes
        self.df['cluster'] = clusters
        if self.data is not None:
            self.data['cluster'] = clusters
            self.data['cluster_label'] = self.data['cluster'].map(self.cluster_label_map)
        
        return clusters
    
    def visualize_pca_clusters(self):
        """Visualise les clusters sur les composantes PCA"""
        if self.kmeans_model is None:
            self.perform_kmeans()
        
        if self.X_pca is None:
            self.perform_pca(n_components=2)
        
        # Définir les compétences (sans la colonne cluster)
        skill_cols = [col for col in self.df.columns if col != 'cluster']
        X = self.df[skill_cols]
        
        # Transformer les données avec PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Transformer les centroids
        centroids = self.kmeans_model.cluster_centers_
        centroids_pca = pca.transform(centroids)
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot des points
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=self.df['cluster'],
            cmap='tab10',
            s=5,
            alpha=0.6
        )
        
        # Ajouter les centres des clusters
        plt.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            c='black',
            s=220,
            marker='X',
            label='Centroïdes',
            edgecolors='white',
            linewidth=2
        )
        
        # Étiqueter les centroids
        for i, (x, y) in enumerate(centroids_pca):
            plt.text(x, y, f'C{i}', fontsize=12, weight='bold', color='white',
                    ha='center', va='center')
        
        plt.xlabel('Composante Principale 1', fontsize=12)
        plt.ylabel('Composante Principale 2', fontsize=12)
        plt.title(f'Visualisation PCA des Clusters de Compétences (k={self.optimal_k})', 
                 fontsize=14, fontweight='bold')
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
    
    def visualize_cluster_distribution(self):
        """Visualise la distribution des clusters"""
        if 'cluster' not in self.data.columns:
            self.perform_kmeans()
        
        cluster_counts = self.data['cluster'].value_counts().sort_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Diagramme circulaire
        axes[0].pie(cluster_counts.values, 
                   labels=[f'Cluster {i}' for i in range(self.optimal_k)],
                   autopct='%1.1f%%', 
                   colors=colors, 
                   startangle=90,
                   textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[0].set_title('Distribution des Clusters', fontsize=14, fontweight='bold')
        
        # Diagramme en barres
        bars = axes[1].bar(range(self.optimal_k), cluster_counts.values, 
                          color=colors, edgecolor='black', linewidth=2)
        axes[1].set_xlabel('Cluster', fontsize=12)
        axes[1].set_ylabel('Nombre de Jobs', fontsize=12)
        axes[1].set_title('Jobs par Cluster', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(self.optimal_k))
        axes[1].set_xticklabels([f'Cluster {i}' for i in range(self.optimal_k)])
        
        # Ajouter les valeurs sur les barres
        for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{count:,}\n({count/len(self.data)*100:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_skills_distribution_by_cluster(self):
        """Visualise la distribution du nombre de compétences par cluster"""
        if 'cluster' not in self.data.columns:
            self.perform_kmeans()
        
        # Calculer le nombre de compétences par job
        if 'num_skills' not in self.data.columns:
            skill_cols = [col for col in self.df.columns if col != 'cluster']
            self.data['num_skills'] = self.df[skill_cols].sum(axis=1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        colors = ['red', 'green', 'blue']
        
        for i in range(self.optimal_k):
            cluster_data = self.data[self.data['cluster'] == i]['num_skills']
            
            axes[i].hist(cluster_data, bins=20, color=colors[i], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Cluster {i}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Nombre de Compétences', fontsize=10)
            axes[i].set_ylabel('Nombre de Jobs', fontsize=10)
            
            # Ajouter la moyenne
            mean_val = cluster_data.mean()
            axes[i].axvline(mean_val, color='black', linestyle='--', linewidth=2,
                           label=f'Moyenne: {mean_val:.1f}')
            axes[i].legend()
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def get_top_skills_by_cluster(self, top_n=10):
        """Obtient les top N compétences pour chaque cluster"""
        if 'cluster' not in self.data.columns:
            self.perform_kmeans()
        
        skill_cols = [col for col in self.df.columns if col != 'cluster']
        
        top_skills_dict = {}
        
        for cluster_id in range(self.optimal_k):
            cluster_jobs = self.data[self.data['cluster'] == cluster_id]
            
            # Calculer le pourcentage de jobs avec chaque compétence
            skill_percentages = {}
            for skill in skill_cols:
                pct = (self.df.loc[cluster_jobs.index, skill].sum() / len(cluster_jobs)) * 100
                skill_percentages[skill] = pct
            
            # Obtenir les top N
            top_skills = sorted(skill_percentages.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_skills_dict[cluster_id] = top_skills
        
        return top_skills_dict
    
    def visualize_top_skills_by_cluster(self, top_n=10):
        """Visualise les top compétences pour chaque cluster"""
        top_skills_dict = self.get_top_skills_by_cluster(top_n)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for cluster_id in range(self.optimal_k):
            top_skills = top_skills_dict[cluster_id]
            skills = [s[0] for s in top_skills]
            percentages = [s[1] for s in top_skills]
            
            axes[cluster_id].barh(skills, percentages, color=colors_bar[cluster_id])
            axes[cluster_id].set_title(f'Cluster {cluster_id}: Top {top_n} Compétences', 
                                      fontweight='bold', fontsize=12)
            axes[cluster_id].set_xlabel('% de Jobs', fontsize=10)
            axes[cluster_id].invert_yaxis()
            
            # Ajouter les pourcentages
            for i, v in enumerate(percentages):
                axes[cluster_id].text(v + 1, i, f'{v:.0f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_key_skills_comparison(self, key_skills=None):
        """Visualise la comparaison des compétences clés entre les clusters"""
        if 'cluster' not in self.data.columns:
            self.perform_kmeans()
        
        skill_cols = [col for col in self.df.columns if col != 'cluster']
        
        if key_skills is None:
            key_skills = ['python', 'sql', 'excel', 'r', 'spark', 'aws',
                         'tableau', 'power bi', 'java', 'hadoop']
        
        # Filtrer uniquement les compétences disponibles
        key_skills = [s for s in key_skills if s in skill_cols]
        
        if not key_skills:
            return None
        
        # Calculer les pourcentages
        comparison = []
        for skill in key_skills:
            for cluster_id in range(self.optimal_k):
                cluster_jobs = self.data[self.data['cluster'] == cluster_id]
                pct = (self.df.loc[cluster_jobs.index, skill].sum() / len(cluster_jobs)) * 100
                comparison.append({
                    'Skill': skill,
                    'Cluster': cluster_id,
                    'Percentage': pct
                })
        
        comparison_df = pd.DataFrame(comparison)
        
        # Créer un graphique en barres groupées
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(key_skills))
        width = 0.25
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i in range(self.optimal_k):
            cluster_data = comparison_df[comparison_df['Cluster'] == i]
            values = cluster_data['Percentage'].values
            plt.bar(x + i*width, values, width, label=f'Cluster {i}', color=colors_bar[i])
        
        plt.xlabel('Compétence', fontsize=12)
        plt.ylabel('% de Jobs', fontsize=12)
        plt.title('Comparaison des Compétences Clés entre les Clusters', 
                 fontsize=14, fontweight='bold')
        plt.xticks(x + width, key_skills, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_radar_chart(self, skills=None):
        """Crée un radar chart pour comparer les profils de compétences"""
        if 'cluster' not in self.data.columns:
            self.perform_kmeans()
        
        skill_cols = [col for col in self.df.columns if col != 'cluster']
        
        if skills is None:
            skills = ['python', 'sql', 'r', 'excel', 'aws', 'spark']
        
        # Filtrer uniquement les compétences disponibles
        skills = [s for s in skills if s in skill_cols]
        
        if not skills or len(skills) < 3:
            return None
        
        # Calculer les taux moyens par cluster
        cluster_skill_rates = self.data.groupby('cluster')[skills].mean()
        
        # Nombre de variables
        categories = list(cluster_skill_rates.columns)
        N = len(categories)
        
        # Calculer les angles
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Tracer chaque cluster
        for cluster_id, row in cluster_skill_rates.iterrows():
            values = row.values.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster_id}', 
                   color=colors_radar[cluster_id])
            ax.fill(angles, values, alpha=0.15, color=colors_radar[cluster_id])
        
        # Ajouter les labels
        plt.xticks(angles[:-1], categories, size=12)
        ax.set_rlabel_position(30)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=10)
        plt.ylim(0, 1)
        plt.title("Profils de Compétences par Cluster (Radar Chart)", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_job_title_distribution(self):
        """Visualise la distribution des titres de jobs par cluster"""
        if 'cluster' not in self.data.columns:
            self.perform_kmeans()
        
        if 'job_title_short' not in self.data.columns:
            return None
        
        # Créer une cross-tab normalisée
        job_cluster_dist = pd.crosstab(
            self.data['job_title_short'],
            self.data['cluster'],
            normalize='index'
        )
        
        plt.figure(figsize=(12, 8))
        job_cluster_dist.plot(
            kind='bar',
            stacked=True,
            color=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        
        plt.title("Distribution des Titres de Jobs par Cluster", 
                 fontsize=14, fontweight='bold')
        plt.xlabel("Titre du Job", fontsize=12)
        plt.ylabel("Proportion", fontsize=12)
        plt.legend(title="Cluster", labels=[f'Cluster {i}' for i in range(self.optimal_k)])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def visualize_skill_heatmap(self):
        """Visualise une heatmap des compétences par cluster"""
        if 'cluster' not in self.data.columns:
            self.perform_kmeans()
        
        skill_cols = [col for col in self.df.columns if col != 'cluster']
        
        # Sélectionner les compétences principales
        main_skills = ['airflow', 'aws', 'azure', 'databricks', 'docker', 'excel', 'gcp', 'git',
                      'go', 'hadoop', 'java', 'kafka', 'kubernetes', 'nosql', 'oracle',
                      'power bi', 'python', 'r', 'sas', 'scala', 'snowflake', 'spark',
                      'sql', 'sql server', 'tableau']
        
        # Filtrer uniquement les compétences disponibles
        available_skills = [s for s in main_skills if s in skill_cols]
        
        if not available_skills:
            available_skills = skill_cols[:25]  # Prendre les 25 premières si aucune correspondance
        
        # Calculer la présence moyenne de chaque compétence par cluster
        skill_cluster_matrix = (
            self.data
            .groupby('cluster')[available_skills]
            .mean()
        )
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(
            skill_cluster_matrix,
            cmap='YlOrRd',
            linewidths=0.5,
            annot=False,
            cbar_kws={'label': 'Proportion de Jobs'}
        )
        
        plt.title("Heatmap des Compétences par Cluster", fontsize=14, fontweight='bold')
        plt.xlabel("Compétence", fontsize=12)
        plt.ylabel("Cluster", fontsize=12)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def get_cluster_summary(self):
        """Génère un résumé détaillé de chaque cluster"""
        if 'cluster' not in self.data.columns:
            self.perform_kmeans()
        
        skill_cols = [col for col in self.df.columns if col != 'cluster']
        
        summaries = {}
        
        for cluster_id in range(self.optimal_k):
            cluster_df = self.data[self.data['cluster'] == cluster_id]
            
            # Calculer le nombre moyen de compétences
            if 'num_skills' not in self.data.columns:
                self.data['num_skills'] = self.df[skill_cols].sum(axis=1)
            
            avg_skills = cluster_df['num_skills'].mean()
            
            # Top compétences
            skill_presence = {}
            for skill in skill_cols:
                pct = (self.df.loc[cluster_df.index, skill].sum() / len(cluster_df)) * 100
                skill_presence[skill] = pct
            
            top_skills = sorted(skill_presence.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Titres de jobs dominants si disponible
            dominant_titles = []
            if 'job_title_short' in self.data.columns:
                title_dist = (
                    cluster_df['job_title_short']
                    .value_counts(normalize=True)
                    .head(5) * 100
                )
                dominant_titles = [(title, pct) for title, pct in title_dist.items()]
            
            summaries[cluster_id] = {
                'label': self.cluster_label_map.get(cluster_id, f'Cluster {cluster_id}'),
                'size': len(cluster_df),
                'percentage': round(len(cluster_df) / len(self.data) * 100, 2),
                'avg_skills': round(avg_skills, 2),
                'top_skills': top_skills,
                'dominant_titles': dominant_titles
            }
        
        return summaries
    
    def predict_cluster(self, skills_dict):
        """
        Prédit le cluster pour un ensemble de compétences donné
        skills_dict: dictionnaire {skill_name: 0 ou 1}
        """
        if self.kmeans_model is None:
            self.perform_kmeans()
        
        skill_cols = [col for col in self.df.columns if col != 'cluster']
        
        # Créer un vecteur de compétences
        skills_vector = []
        for skill in skill_cols:
            skills_vector.append(skills_dict.get(skill, 0))
        
        # Prédire le cluster
        cluster = self.kmeans_model.predict([skills_vector])[0]
        cluster_label = self.cluster_label_map.get(cluster, f'Cluster {cluster}')
        
        return int(cluster), cluster_label
