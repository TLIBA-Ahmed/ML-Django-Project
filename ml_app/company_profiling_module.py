"""
Module de Profiling et Segmentation des Entreprises/Emplois (DS1)
Analyse les profils d'entreprises selon le secteur, la taille et l'adoption IA
Clustering K-Means pour segmentation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io
import base64
import os
from django.conf import settings


class CompanyProfilingModel:
    """Modèle de profiling et segmentation des entreprises"""
    
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(settings.BASE_DIR, 'data_prep_with_clusters.csv')
        self.csv_path = csv_path
        self.df = None
        self.df_encoded = None
        self.kmeans_model = None
        self.scaler = None
        self.pca = None
        self.onehot_encoder = None
        
    def load_data(self):
        """Charge les données depuis le CSV"""
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Vérifier les colonnes nécessaires
            required_cols = ['sector_group', 'company_size', 'ai_profile', 
                           'salary_usd', 'nb_skills', 'year_experience']
            
            for col in required_cols:
                if col not in self.df.columns:
                    raise ValueError(f"Colonne manquante: {col}")
            
            # Nettoyer les données
            self.df['sector_group'] = self.df['sector_group'].fillna('Unknown')
            self.df['company_size'] = self.df['company_size'].fillna('Unknown')
            self.df['ai_profile'] = self.df['ai_profile'].fillna('Low AI')
            
            return True
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return False
    
    def prepare_features(self):
        """Prépare les features pour le clustering"""
        try:
            # Colonnes catégorielles pour le clustering
            cat_cols = ['sector_group', 'company_size', 'ai_profile']
            
            # One-Hot Encoding
            self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', 
                                               handle_unknown='ignore')
            encoded_features = self.onehot_encoder.fit_transform(self.df[cat_cols])
            
            # Créer DataFrame encodé
            feature_names = self.onehot_encoder.get_feature_names_out(cat_cols)
            self.df_encoded = pd.DataFrame(encoded_features, columns=feature_names)
            self.df_encoded = self.df_encoded.dropna()
            
            return True
        except Exception as e:
            print(f"Erreur lors de la préparation: {e}")
            return False
    
    def train_kmeans(self, n_clusters=4):
        """Entraîne le modèle K-Means"""
        try:
            if self.df_encoded is None:
                return None
            
            # Standardisation
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.df_encoded)
            
            # K-Means
            self.kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', 
                                      random_state=42, n_init=10)
            clusters = self.kmeans_model.fit_predict(X_scaled)
            
            # Ajouter les clusters au DataFrame
            self.df['cluster_id'] = np.nan
            self.df.loc[self.df_encoded.index, 'cluster_id'] = clusters
            
            # Calculer le score silhouette
            silhouette_avg = silhouette_score(X_scaled, clusters)
            
            return {
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg,
                'inertia': self.kmeans_model.inertia_
            }
        except Exception as e:
            print(f"Erreur lors du clustering: {e}")
            return None
    
    def find_optimal_k(self, max_k=10):
        """Trouve le nombre optimal de clusters (méthode du coude)"""
        try:
            if self.df_encoded is None:
                return None
            
            X_scaled = self.scaler.fit_transform(self.df_encoded)
            
            wcss = []
            silhouette_scores = []
            k_range = range(2, max_k + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, init='k-means++', 
                              random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)
                
                silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(silhouette_avg)
            
            return {
                'k_values': list(k_range),
                'wcss': wcss,
                'silhouette_scores': silhouette_scores
            }
        except Exception as e:
            print(f"Erreur lors de l'optimisation: {e}")
            return None
    
    def get_cluster_profiles(self):
        """Obtient les profils des clusters"""
        try:
            df_clean = self.df.dropna(subset=['cluster_id'])
            
            profiles = df_clean.groupby('cluster_id').agg({
                'salary_usd': 'mean',
                'nb_skills': 'mean',
                'year_experience': 'mean',
                'company_size': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
                'sector_group': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
                'ai_profile': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Low AI',
                'cluster_id': 'count'
            }).rename(columns={'cluster_id': 'count'}).reset_index()
            
            return profiles
        except Exception as e:
            print(f"Erreur lors du profiling: {e}")
            return None
    
    def predict_cluster(self, sector_group, company_size, ai_profile):
        """Prédit le cluster pour de nouvelles données"""
        try:
            if self.kmeans_model is None or self.onehot_encoder is None:
                return None
            
            # Créer DataFrame avec les inputs
            input_data = pd.DataFrame({
                'sector_group': [sector_group],
                'company_size': [company_size],
                'ai_profile': [ai_profile]
            })
            
            # Encoder
            encoded = self.onehot_encoder.transform(input_data)
            
            # Standardiser
            scaled = self.scaler.transform(encoded)
            
            # Prédire
            cluster = self.kmeans_model.predict(scaled)[0]
            
            return int(cluster)
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            return None
    
    def plot_elbow_method(self):
        """Génère le graphique de la méthode du coude"""
        try:
            optimal_k_data = self.find_optimal_k()
            if optimal_k_data is None:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # WCSS
            ax1.plot(optimal_k_data['k_values'], optimal_k_data['wcss'], 
                    marker='o', linestyle='--', color='blue')
            ax1.set_title('Méthode du Coude (WCSS)')
            ax1.set_xlabel('Nombre de clusters (K)')
            ax1.set_ylabel('WCSS')
            ax1.grid(True)
            
            # Silhouette
            ax2.plot(optimal_k_data['k_values'], optimal_k_data['silhouette_scores'], 
                    marker='o', linestyle='--', color='green')
            ax2.set_title('Score Silhouette')
            ax2.set_xlabel('Nombre de clusters (K)')
            ax2.set_ylabel('Score Silhouette')
            ax2.grid(True)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'data:image/png;base64,{image_base64}'
        except Exception as e:
            print(f"Erreur lors de la visualisation: {e}")
            return None
    
    def plot_pca_clusters(self):
        """Visualise les clusters avec PCA"""
        try:
            if self.df_encoded is None or self.kmeans_model is None:
                return None
            
            # PCA
            self.pca = PCA(n_components=2)
            X_scaled = self.scaler.transform(self.df_encoded)
            X_pca = self.pca.fit_transform(X_scaled)
            
            # DataFrame pour visualisation
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'cluster_id': self.df.loc[self.df_encoded.index, 'cluster_id'].values,
                'ai_profile': self.df.loc[self.df_encoded.index, 'ai_profile'].values
            })
            
            explained = self.pca.explained_variance_ratio_ * 100
            
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='PC1', y='PC2', hue='cluster_id', style='ai_profile',
                          data=pca_df, palette='deep', legend='full', alpha=0.7, s=80)
            plt.title(f'Segmentation des Entreprises (K-Means)\nVariance expliquée: PC1={explained[0]:.1f}% / PC2={explained[1]:.1f}%')
            plt.xlabel(f'Composante Principale 1 ({explained[0]:.1f}%)')
            plt.ylabel(f'Composante Principale 2 ({explained[1]:.1f}%)')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Cluster / Profil IA', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'data:image/png;base64,{image_base64}'
        except Exception as e:
            print(f"Erreur lors de la visualisation PCA: {e}")
            return None
    
    def plot_cluster_composition(self):
        """Visualise la composition des clusters"""
        try:
            df_clean = self.df.dropna(subset=['cluster_id'])
            
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            
            # AI Profile par cluster
            sns.countplot(data=df_clean, x='cluster_id', hue='ai_profile', 
                         palette='viridis', ax=axes[0])
            axes[0].set_title('Profil IA par Cluster')
            axes[0].set_xlabel('Cluster')
            axes[0].set_ylabel('Nombre')
            
            # Company Size par cluster
            sns.countplot(data=df_clean, x='cluster_id', hue='company_size', 
                         palette='magma', ax=axes[1])
            axes[1].set_title('Taille Entreprise par Cluster')
            axes[1].set_xlabel('Cluster')
            axes[1].set_ylabel('Nombre')
            
            # Top 5 secteurs
            top_sectors = df_clean['sector_group'].value_counts().nlargest(5).index
            df_top_sectors = df_clean[df_clean['sector_group'].isin(top_sectors)]
            sns.countplot(data=df_top_sectors, x='cluster_id', hue='sector_group', 
                         palette='Set2', ax=axes[2])
            axes[2].set_title('Top 5 Secteurs par Cluster')
            axes[2].set_xlabel('Cluster')
            axes[2].set_ylabel('Nombre')
            axes[2].legend(title='Secteur', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'data:image/png;base64,{image_base64}'
        except Exception as e:
            print(f"Erreur lors de la visualisation: {e}")
            return None
    
    def plot_profiling_bars(self):
        """Graphiques de profiling par taille et secteur"""
        try:
            df_clean = self.df.dropna(subset=['cluster_id'])
            
            # Profiling par taille et AI
            company_profile = df_clean.groupby(['company_size', 'ai_profile'])[['salary_usd', 'nb_skills']].mean().reset_index()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Salaire moyen
            sns.barplot(data=company_profile, x='company_size', y='salary_usd', 
                       hue='ai_profile', palette='viridis', ax=ax1)
            ax1.set_title('Salaire Moyen par Taille et Profil IA')
            ax1.set_xlabel('Taille Entreprise')
            ax1.set_ylabel('Salaire USD')
            ax1.tick_params(axis='x', rotation=45)
            
            # Compétences moyennes
            sns.barplot(data=company_profile, x='company_size', y='nb_skills', 
                       hue='ai_profile', palette='magma', ax=ax2)
            ax2.set_title('Compétences Moyennes par Taille et Profil IA')
            ax2.set_xlabel('Taille Entreprise')
            ax2.set_ylabel('Nombre de Compétences')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'data:image/png;base64,{image_base64}'
        except Exception as e:
            print(f"Erreur lors de la visualisation: {e}")
            return None
