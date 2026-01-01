"""
Tests pour le module Job Skill Clustering
"""
import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_project.settings')
django.setup()

from ml_app.job_skill_clustering_module import JobSkillClusteringModel
import pandas as pd
import numpy as np

def test_data_loading():
    """Test de chargement des données"""
    print("="*80)
    print("TEST 1: Chargement des données")
    print("="*80)
    
    try:
        model = JobSkillClusteringModel()
        df, data = model.load_data()
        
        print(f"✓ Données chargées avec succès")
        print(f"  - Shape df (skills): {df.shape}")
        print(f"  - Shape data (full): {data.shape}")
        print(f"  - Colonnes df (5 premières): {df.columns[:5].tolist()}")
        
        return model
    except Exception as e:
        print(f"✗ Erreur: {str(e)}")
        return None

def test_preprocessing(model):
    """Test du prétraitement"""
    print("\n" + "="*80)
    print("TEST 2: Prétraitement des données")
    print("="*80)
    
    try:
        X_scaled = model.preprocess_data()
        print(f"✓ Prétraitement réussi")
        print(f"  - Shape données normalisées: {X_scaled.shape}")
        print(f"  - Type: {type(X_scaled)}")
        print(f"  - Min/Max: {X_scaled.min():.3f} / {X_scaled.max():.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {str(e)}")
        return False

def test_pca(model):
    """Test de la PCA"""
    print("\n" + "="*80)
    print("TEST 3: Réduction de dimensionnalité (PCA)")
    print("="*80)
    
    try:
        X_pca, variance = model.perform_pca(n_components=2)
        print(f"✓ PCA réussie")
        print(f"  - Shape données PCA: {X_pca.shape}")
        print(f"  - Variance expliquée: {variance:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {str(e)}")
        return False

def test_elbow_method(model):
    """Test de la méthode du coude"""
    print("\n" + "="*80)
    print("TEST 4: Méthode du coude")
    print("="*80)
    
    try:
        _, inertias = model.elbow_method(k_range=range(2, 6))
        print(f"✓ Méthode du coude réussie")
        print(f"  - Nombre d'inertias calculées: {len(inertias)}")
        print(f"  - Inertias: {[f'{i:.2f}' for i in inertias]}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {str(e)}")
        return False

def test_silhouette(model):
    """Test du score de silhouette"""
    print("\n" + "="*80)
    print("TEST 5: Score de Silhouette")
    print("="*80)
    
    try:
        _, scores = model.silhouette_analysis(k_range=range(2, 6), sample_size=500)
        print(f"✓ Analyse silhouette réussie")
        print(f"  - Nombre de scores: {len(scores)}")
        print(f"  - Scores: {[f'{s:.3f}' for s in scores]}")
        print(f"  - Meilleur k: {scores.index(max(scores)) + 2}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {str(e)}")
        return False

def test_kmeans_clustering(model):
    """Test du clustering K-means"""
    print("\n" + "="*80)
    print("TEST 6: Clustering K-means (k=3)")
    print("="*80)
    
    try:
        clusters = model.perform_kmeans(n_clusters=3)
        print(f"✓ Clustering réussi")
        print(f"  - Nombre de jobs: {len(clusters)}")
        print(f"  - Clusters uniques: {np.unique(clusters)}")
        
        # Distribution des clusters
        unique, counts = np.unique(clusters, return_counts=True)
        print(f"\n  Distribution:")
        for cluster_id, count in zip(unique, counts):
            pct = (count / len(clusters)) * 100
            print(f"    Cluster {cluster_id}: {count:,} jobs ({pct:.2f}%)")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {str(e)}")
        return False

def test_cluster_summary(model):
    """Test du résumé des clusters"""
    print("\n" + "="*80)
    print("TEST 7: Résumé des clusters")
    print("="*80)
    
    try:
        summary = model.get_cluster_summary()
        print(f"✓ Résumé généré avec succès")
        
        for cluster_id, info in summary.items():
            print(f"\n  Cluster {cluster_id}: {info['label']}")
            print(f"    - Taille: {info['size']} jobs ({info['percentage']}%)")
            print(f"    - Compétences moyennes: {info['avg_skills']}")
            print(f"    - Top 3 compétences:")
            for i, (skill, pct) in enumerate(info['top_skills'][:3], 1):
                print(f"      {i}. {skill}: {pct:.1f}%")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {str(e)}")
        return False

def test_prediction(model):
    """Test de prédiction"""
    print("\n" + "="*80)
    print("TEST 8: Prédiction de cluster")
    print("="*80)
    
    # Simuler différents profils de compétences
    test_profiles = [
        {
            'name': 'Business Analyst',
            'skills': {'python': 1, 'sql': 1, 'excel': 1, 'tableau': 1, 'power bi': 1}
        },
        {
            'name': 'Data Engineer',
            'skills': {'python': 1, 'sql': 1, 'spark': 1, 'aws': 1, 'kafka': 1, 'docker': 1}
        },
        {
            'name': 'Data Scientist',
            'skills': {'python': 1, 'r': 1, 'sql': 1, 'spark': 1}
        }
    ]
    
    try:
        for profile in test_profiles:
            # Créer un dictionnaire complet avec toutes les compétences
            all_skills = [col for col in model.df.columns if col != 'cluster']
            skills_dict = {skill: profile['skills'].get(skill, 0) for skill in all_skills}
            
            cluster, label = model.predict_cluster(skills_dict)
            print(f"\n  Profil: {profile['name']}")
            print(f"  Compétences: {', '.join(profile['skills'].keys())}")
            print(f"  → Cluster prédit: {cluster} - {label}")
        
        print(f"\n✓ Prédictions réussies")
        return True
    except Exception as e:
        print(f"✗ Erreur: {str(e)}")
        return False

def test_visualizations(model):
    """Test de génération des visualisations"""
    print("\n" + "="*80)
    print("TEST 9: Génération des visualisations")
    print("="*80)
    
    visualizations = {
        'PCA Clusters': lambda: model.visualize_pca_clusters(),
        'Distribution': lambda: model.visualize_cluster_distribution(),
        'Skills Distribution': lambda: model.visualize_skills_distribution_by_cluster(),
        'Top Skills': lambda: model.visualize_top_skills_by_cluster(),
        'Skills Comparison': lambda: model.visualize_key_skills_comparison(),
        'Radar Chart': lambda: model.visualize_radar_chart(),
        'Heatmap': lambda: model.visualize_skill_heatmap(),
    }
    
    success_count = 0
    for viz_name, viz_func in visualizations.items():
        try:
            result = viz_func()
            if result is not None:
                print(f"  ✓ {viz_name}: OK")
                success_count += 1
            else:
                print(f"  ⚠ {viz_name}: None (peut-être normal)")
        except Exception as e:
            print(f"  ✗ {viz_name}: {str(e)}")
    
    print(f"\n✓ {success_count}/{len(visualizations)} visualisations générées")
    return success_count > 0

def test_cache_system(model):
    """Test du système de cache"""
    print("\n" + "="*80)
    print("TEST 10: Système de cache")
    print("="*80)
    
    try:
        # Sauvegarder
        model.save_models()
        print(f"✓ Modèles sauvegardés")
        
        # Créer une nouvelle instance et charger
        new_model = JobSkillClusteringModel()
        new_model.load_data()
        new_model.preprocess_data()
        success = new_model.load_models()
        
        if success:
            print(f"✓ Modèles chargés depuis le cache")
            print(f"  - K-means: {'Chargé' if new_model.kmeans_model is not None else 'Non chargé'}")
            print(f"  - PCA: {'Chargé' if new_model.pca_model is not None else 'Non chargé'}")
            print(f"  - Scaler: {'Chargé' if new_model.scaler is not None else 'Non chargé'}")
        else:
            print(f"⚠ Cache non disponible ou erreur de chargement")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {str(e)}")
        return False

def run_all_tests():
    """Exécuter tous les tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "JOB SKILL CLUSTERING - TESTS" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    results = []
    
    # Test 1: Chargement
    model = test_data_loading()
    results.append(('Chargement des données', model is not None))
    
    if model is None:
        print("\n✗ Tests arrêtés car le chargement des données a échoué")
        return
    
    # Tests suivants
    results.append(('Prétraitement', test_preprocessing(model)))
    results.append(('PCA', test_pca(model)))
    results.append(('Méthode du coude', test_elbow_method(model)))
    results.append(('Score de Silhouette', test_silhouette(model)))
    results.append(('K-means Clustering', test_kmeans_clustering(model)))
    results.append(('Résumé des clusters', test_cluster_summary(model)))
    results.append(('Prédiction', test_prediction(model)))
    results.append(('Visualisations', test_visualizations(model)))
    results.append(('Système de cache', test_cache_system(model)))
    
    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} - {test_name}")
    
    print("\n" + "-"*80)
    print(f"Résultat: {passed}/{total} tests réussis ({passed/total*100:.1f}%)")
    print("="*80 + "\n")
    
    if passed == total:
        print("🎉 Tous les tests sont passés avec succès!")
    elif passed >= total * 0.8:
        print("⚠ La plupart des tests sont passés, mais certains nécessitent de l'attention")
    else:
        print("❌ Plusieurs tests ont échoué, veuillez vérifier la configuration")

if __name__ == '__main__':
    run_all_tests()
