"""
Script de vérification rapide pour l'intégration du Job Skill Clustering
"""
import os
import sys

def check_file_exists(filepath, description):
    """Vérifie si un fichier existe"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}")
    if not exists:
        print(f"   Manquant: {filepath}")
    return exists

def check_string_in_file(filepath, search_string, description):
    """Vérifie si une chaîne existe dans un fichier"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            found = search_string in content
            status = "✓" if found else "✗"
            print(f"{status} {description}")
            return found
    except Exception as e:
        print(f"✗ {description} (Erreur: {e})")
        return False

def main():
    print("\n" + "="*80)
    print("VÉRIFICATION DE L'INTÉGRATION - JOB SKILL CLUSTERING")
    print("="*80 + "\n")
    
    base_dir = r"c:\Users\Tliba\Desktop\integration ML\ml_django_project"
    parent_dir = r"c:\Users\Tliba\Desktop\integration ML"
    
    checks_passed = 0
    total_checks = 0
    
    # 1. Vérifier les fichiers de module
    print("1. FICHIERS DU MODULE")
    print("-" * 80)
    
    files_to_check = [
        (os.path.join(base_dir, "ml_app", "job_skill_clustering_module.py"), 
         "Module principal"),
        (os.path.join(base_dir, "templates", "ml_app", "job_skill_clustering_analysis.html"),
         "Template d'analyse"),
        (os.path.join(base_dir, "templates", "ml_app", "job_skill_clustering_predict.html"),
         "Template de prédiction"),
        (os.path.join(base_dir, "JOB_SKILL_CLUSTERING_README.md"),
         "Documentation README"),
        (os.path.join(base_dir, "test_job_skill_clustering.py"),
         "Fichier de tests"),
        (os.path.join(base_dir, "INTEGRATION_JOB_SKILL_CLUSTERING.md"),
         "Document d'intégration"),
    ]
    
    for filepath, desc in files_to_check:
        if check_file_exists(filepath, desc):
            checks_passed += 1
        total_checks += 1
    
    # 2. Vérifier les fichiers de données
    print("\n2. FICHIERS DE DONNÉES")
    print("-" * 80)
    
    data_files = [
        (os.path.join(parent_dir, "processed_data_jobs.csv"),
         "Fichier de compétences (REQUIS)"),
        (os.path.join(parent_dir, "df_final_ml.csv"),
         "Fichier de données complètes (OPTIONNEL)"),
    ]
    
    for filepath, desc in data_files:
        if check_file_exists(filepath, desc):
            checks_passed += 1
        total_checks += 1
    
    # 3. Vérifier les modifications dans views.py
    print("\n3. MODIFICATIONS DE VIEWS.PY")
    print("-" * 80)
    
    views_file = os.path.join(base_dir, "ml_app", "views.py")
    views_checks = [
        ("from .job_skill_clustering_module import JobSkillClusteringModel",
         "Import du module"),
        ("def job_skill_clustering_analysis(request):",
         "Vue d'analyse"),
        ("def job_skill_clustering_predict(request):",
         "Vue de prédiction"),
    ]
    
    for search_str, desc in views_checks:
        if check_string_in_file(views_file, search_str, desc):
            checks_passed += 1
        total_checks += 1
    
    # 4. Vérifier les URLs
    print("\n4. CONFIGURATION DES URLs")
    print("-" * 80)
    
    urls_file = os.path.join(base_dir, "ml_app", "urls.py")
    url_checks = [
        ("job-skill-clustering/", "URL d'analyse"),
        ("job_skill_clustering_analysis", "Nom de la vue d'analyse"),
        ("job_skill_clustering_predict", "Nom de la vue de prédiction"),
    ]
    
    for search_str, desc in url_checks:
        if check_string_in_file(urls_file, search_str, desc):
            checks_passed += 1
        total_checks += 1
    
    # 5. Vérifier la page d'accueil
    print("\n5. MISE À JOUR DE LA PAGE D'ACCUEIL")
    print("-" * 80)
    
    home_file = os.path.join(base_dir, "templates", "ml_app", "home.html")
    home_checks = [
        ("Clustering par Compétences", "Titre de la carte"),
        ("job_skill_clustering_analysis", "Lien vers l'analyse"),
        ("job_skill_clustering_predict", "Lien vers la prédiction"),
    ]
    
    for search_str, desc in home_checks:
        if check_string_in_file(home_file, search_str, desc):
            checks_passed += 1
        total_checks += 1
    
    # 6. Vérifier la structure du module
    print("\n6. CONTENU DU MODULE")
    print("-" * 80)
    
    module_file = os.path.join(base_dir, "ml_app", "job_skill_clustering_module.py")
    module_checks = [
        ("class JobSkillClusteringModel:", "Classe principale"),
        ("def load_data(self):", "Méthode load_data"),
        ("def preprocess_data(self):", "Méthode preprocess_data"),
        ("def perform_kmeans(self", "Méthode perform_kmeans"),
        ("def predict_cluster(self", "Méthode predict_cluster"),
        ("def visualize_pca_clusters(self):", "Méthode de visualisation PCA"),
        ("self.optimal_k = 3", "K optimal = 3"),
    ]
    
    for search_str, desc in module_checks:
        if check_string_in_file(module_file, search_str, desc):
            checks_passed += 1
        total_checks += 1
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)
    
    percentage = (checks_passed / total_checks) * 100
    print(f"\nRésultat: {checks_passed}/{total_checks} vérifications réussies ({percentage:.1f}%)")
    
    if checks_passed == total_checks:
        print("\n🎉 Intégration complète ! Tous les fichiers sont en place.")
        print("\nProchaines étapes:")
        print("1. Lancer le serveur: python manage.py runserver")
        print("2. Se connecter en tant qu'admin")
        print("3. Accéder à /job-skill-clustering/ pour l'analyse")
        print("4. Tester la prédiction sur /job-skill-clustering/predict/")
    elif checks_passed >= total_checks * 0.8:
        print("\n⚠️  Presque terminé ! Quelques éléments manquants.")
        print("Vérifiez les erreurs ci-dessus.")
    else:
        print("\n❌ Plusieurs éléments sont manquants.")
        print("Veuillez revoir l'intégration.")
    
    print("\n" + "="*80 + "\n")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
