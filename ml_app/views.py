from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .models import (SalaryPrediction, JobClusterPrediction, ClusteringResult, 
                     PlatformPrediction, JobTitlePrediction, TimeSeriesPrediction,
                     ExperiencePrediction, EmployeeClusterPrediction,
                     CompanySegmentation, JobAnalysisPrediction)
from .clustering_module import ClusteringAnalysis
from .salary_prediction_module import SalaryPredictionModel
from .job_title_classification_module import JobTitleClassificationModel
from .time_series_forecast_module import TimeSeriesForecastModel
from .experience_prediction_module import ExperiencePredictionModel
from .employee_clustering_module import EmployeeClusteringModel
from .company_profiling_module import CompanyProfilingModel
from .job_analysis_module import JobAnalysisModel
from .chatbot_module import get_chatbot_instance
import json
import os


def home(request):
    """Page d'accueil"""
    return render(request, 'ml_app/home.html')


# Vues d'authentification
def login_view(request):
    """Vue de connexion"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Bienvenue {user.username}!')
            return redirect('ml_app:home')
        else:
            messages.error(request, 'Nom d\'utilisateur ou mot de passe incorrect')
    return render(request, 'ml_app/login.html')


def logout_view(request):
    """Vue de déconnexion"""
    logout(request)
    messages.success(request, 'Vous êtes déconnecté')
    return redirect('ml_app:login')


def register_view(request):
    """Vue d'inscription"""
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        
        if password != password_confirm:
            messages.error(request, 'Les mots de passe ne correspondent pas')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Ce nom d\'utilisateur existe déjà')
        elif User.objects.filter(email=email).exists():
            messages.error(request, 'Cet email est déjà utilisé')
        else:
            try:
                # Créer et sauvegarder l'utilisateur dans la base de données
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save()  # Sauvegarder explicitement
                
                # Connecter l'utilisateur automatiquement après l'inscription
                login(request, user)
                messages.success(request, f'Compte créé avec succès! Vous pouvez maintenant vous reconnecter à tout moment avec votre nom d\'utilisateur et mot de passe.')
                return redirect('ml_app:home')
            except Exception as e:
                messages.error(request, f'Erreur lors de la création du compte: {str(e)}')
    return render(request, 'ml_app/register.html')


def is_admin(user):
    """Vérifie si l'utilisateur est admin"""
    return user.is_staff or user.is_superuser


@user_passes_test(is_admin, login_url='/login/')
def clustering_analysis(request):
    """Page d'analyse de clustering"""
    context = {}
    
    try:
        # Initialiser l'analyse de clustering
        clustering = ClusteringAnalysis()
        clustering.load_data()
        clustering.preprocess_data()
        clustering.perform_pca()
        
        # Générer les visualisations principales
        elbow_plot = clustering.elbow_method()
        clustering.perform_kmeans(n_clusters=4)
        cluster_plot = clustering.visualize_clusters()
        profiles = clustering.cluster_profiles()
        
        # Nouvelles visualisations
        cluster_sizes_plot = clustering.visualize_cluster_sizes()
        cluster_distributions_plot = clustering.visualize_cluster_distributions()
        cluster_comparison_plot = clustering.visualize_cluster_profiles_comparison()
        
        # Sauvegarder le résultat dans la base de données
        ClusteringResult.objects.create(
            n_clusters=4,
            algorithm='K-means'
        )
        
        context = {
            'elbow_plot': elbow_plot,
            'cluster_plot': cluster_plot,
            'cluster_sizes_plot': cluster_sizes_plot,
            'cluster_distributions_plot': cluster_distributions_plot,
            'cluster_comparison_plot': cluster_comparison_plot,
            'profiles': profiles,
            'data_info': {
                'total_jobs': len(clustering.df),
                'features': len(clustering.df.columns)
            }
        }
        
    except Exception as e:
        messages.error(request, f"Erreur lors de l'analyse de clustering: {str(e)}")
        context['error'] = str(e)
    
    return render(request, 'ml_app/clustering_analysis.html', context)


@login_required
def clustering_predict(request):
    """Page de prédiction de cluster pour un job"""
    # Charger les options pour les listes déroulantes
    try:
        clustering = ClusteringAnalysis()
        clustering.load_data()
        
        # Extraire les valeurs uniques pour les listes déroulantes
        job_titles = sorted(clustering.df['job_title'].unique().tolist())
        experience_levels = sorted(clustering.df['experience_level'].unique().tolist())
        employment_types = sorted(clustering.df['employment_type'].unique().tolist())
        company_locations = sorted(clustering.df['company_location'].unique().tolist())
        company_sizes = sorted(clustering.df['company_size'].unique().tolist())
        
        context = {
            'job_titles': job_titles,
            'experience_levels': experience_levels,
            'employment_types': employment_types,
            'company_locations': company_locations,
            'company_sizes': company_sizes
        }
    except Exception as e:
        context = {}
    
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            job_data = {
                'job_title': request.POST.get('job_title'),
                'salary_usd': float(request.POST.get('salary_usd')),
                'experience_level': request.POST.get('experience_level'),
                'employment_type': request.POST.get('employment_type'),
                'company_location': request.POST.get('company_location'),
                'company_size': request.POST.get('company_size'),
                'years_experience': float(request.POST.get('years_experience'))
            }
            
            # Initialiser et préparer le modèle
            clustering = ClusteringAnalysis()
            clustering.load_data()
            clustering.preprocess_data()
            clustering.perform_pca()
            clustering.perform_kmeans(n_clusters=4)
            
            # Faire la prédiction
            predicted_cluster = clustering.predict_cluster(job_data)
            
            # Sauvegarder dans la base de données
            JobClusterPrediction.objects.create(
                user=request.user,
                job_title=job_data['job_title'],
                salary_usd=job_data['salary_usd'],
                experience_level=job_data['experience_level'],
                employment_type=job_data['employment_type'],
                company_location=job_data['company_location'],
                company_size=job_data['company_size'],
                years_experience=job_data['years_experience'],
                predicted_cluster=predicted_cluster
            )
            
            messages.success(request, f"Prédiction réussie! Ce job appartient au Cluster {predicted_cluster}")
            
            context.update({
                'predicted_cluster': predicted_cluster,
                'job_data': job_data
            })
            
            return render(request, 'ml_app/clustering_predict.html', context)
            
        except Exception as e:
            messages.error(request, f"Erreur lors de la prédiction: {str(e)}")
    
    return render(request, 'ml_app/clustering_predict.html', context)


@user_passes_test(is_admin, login_url='/login/')
def salary_analysis(request):
    """Page d'analyse de prédiction de salaire"""
    context = {}
    
    try:
        # Initialiser le modèle de prédiction de salaire
        salary_model = SalaryPredictionModel()
        salary_model.load_data()
        salary_model.preprocess_data()
        salary_model.split_data()
        salary_model.train_models()
        
        # Générer les visualisations principales
        correlation_plot = salary_model.correlation_matrix()
        comparison_plot = salary_model.visualize_comparison()
        results = salary_model.evaluate_models()
        
        # Visualisations pour chaque modèle
        prediction_plots = {}
        for model_name in salary_model.models.keys():
            prediction_plots[model_name] = salary_model.visualize_predictions(model_name)
        
        # Nouvelles visualisations
        salary_dist_plot = salary_model.visualize_salary_distribution()
        salary_by_category_plot = salary_model.visualize_salary_by_category()
        residuals_plot = salary_model.visualize_residuals()
        
        context = {
            'correlation_plot': correlation_plot,
            'comparison_plot': comparison_plot,
            'salary_dist_plot': salary_dist_plot,
            'salary_by_category_plot': salary_by_category_plot,
            'residuals_plot': residuals_plot,
            'results': results,
            'prediction_plots': prediction_plots,
            'best_model': salary_model.best_model_name,
            'data_info': {
                'total_records': len(salary_model.df),
                'features': len(salary_model.df.columns) - 1
            }
        }
        
    except Exception as e:
        messages.error(request, f"Erreur lors de l'analyse de salaire: {str(e)}")
        context['error'] = str(e)
    
    return render(request, 'ml_app/salary_analysis.html', context)


@login_required
def salary_predict(request):
    """Page de prédiction de salaire"""
    # Charger les options pour les listes déroulantes
    try:
        salary_model = SalaryPredictionModel()
        unique_values = salary_model.get_unique_values()
        
        context = {
            'genders': unique_values['genders'],
            'education_levels': unique_values['education_levels'],
            'job_titles': unique_values['job_titles'],
            'model_names': ['Linear Regression', 'Polynomial Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
        }
    except Exception as e:
        messages.warning(request, f"Impossible de charger les options depuis les données: {str(e)}")
        context = {
            'genders': ['Male', 'Female', 'Other'],
            'education_levels': ["Bachelor's", "Master's", 'PhD', 'High School'],
            'job_titles': ['Software Engineer', 'Data Scientist', 'Product Manager', 'Sales Manager'],
            'model_names': ['Linear Regression', 'Polynomial Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
        }
    
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            input_data = {
                'Age': float(request.POST.get('age')),
                'Gender': request.POST.get('gender'),
                'Education Level': request.POST.get('education_level'),
                'Job Title': request.POST.get('job_title'),
                'Years of Experience': float(request.POST.get('years_of_experience'))
            }
            
            model_name = request.POST.get('model_name', None)
            
            # Initialiser et préparer le modèle
            salary_model = SalaryPredictionModel()
            salary_model.load_data()
            salary_model.preprocess_data()
            salary_model.split_data()
            
            # Charger les modèles depuis le cache ou entraîner
            models_loaded = salary_model.load_models()
            if not models_loaded:
                salary_model.train_models()
            
            salary_model.evaluate_models()
            
            # Faire la prédiction
            predicted_salary = salary_model.predict_salary(input_data, model_name)
            
            # Sauvegarder dans la base de données
            SalaryPrediction.objects.create(
                user=request.user,
                age=input_data['Age'],
                years_of_experience=input_data['Years of Experience'],
                gender=input_data['Gender'],
                education_level=input_data['Education Level'],
                job_title=input_data['Job Title'],
                predicted_salary=predicted_salary,
                model_used=model_name or salary_model.best_model_name
            )
            
            messages.success(request, f"Prédiction réussie! Salaire estimé: ${predicted_salary:,.2f}")
            
            # Créer une version avec des clés sans espaces pour le template
            display_data = {
                'Age': input_data['Age'],
                'Gender': input_data['Gender'],
                'Education_Level': input_data['Education Level'],
                'Job_Title': input_data['Job Title'],
                'Years_of_Experience': input_data['Years of Experience']
            }
            
            context.update({
                'predicted_salary': predicted_salary,
                'input_data': display_data,
                'model_used': model_name or salary_model.best_model_name
            })
            
            return render(request, 'ml_app/salary_predict.html', context)
            
        except Exception as e:
            messages.error(request, f"Erreur lors de la prédiction: {str(e)}")
    
    return render(request, 'ml_app/salary_predict.html', context)


@user_passes_test(is_admin, login_url='/login/')
def classification_analysis(request):
    """Page d'analyse de classification des plateformes"""
    context = {}
    
    try:
        from .classification_module import PlatformClassificationModel
        
        # Initialiser le modèle
        classifier = PlatformClassificationModel()
        
        # Toujours charger et préparer les données
        classifier.load_data(sample_size=50000)
        classifier.preprocess_data()
        classifier.prepare_features()
        classifier.split_data()
        
        # Essayer de charger les modèles depuis le cache
        models_loaded = classifier.load_models()
        
        if not models_loaded:
            # Si pas de cache, entraîner les modèles
            classifier.train_models()
        
        # Évaluer les modèles
        results = classifier.evaluate_models()
        
        # Générer les visualisations principales
        comparison_plot = classifier.visualize_comparison()
        confusion_plot = classifier.confusion_matrix_plot()
        
        # Nouvelles visualisations
        platform_dist_plot = classifier.visualize_platform_distribution()
        feature_importance_plot = classifier.visualize_feature_importance()
        classification_report_plot = classifier.visualize_classification_report()
        
        context = {
            'comparison_plot': comparison_plot,
            'confusion_plot': confusion_plot,
            'platform_dist_plot': platform_dist_plot,
            'feature_importance_plot': feature_importance_plot,
            'classification_report_plot': classification_report_plot,
            'results': results,
            'best_model': classifier.best_model_name,
            'data_info': {
                'total_jobs': len(classifier.df_balanced) if classifier.df_balanced is not None else 0,
                'platforms': len(classifier.label_encoder.classes_) if classifier.label_encoder else 0,
                'features': len(classifier.feature_names) if classifier.feature_names else 0
            }
        }
        
    except Exception as e:
        messages.error(request, f"Erreur lors de l'analyse de classification: {str(e)}")
        context['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return render(request, 'ml_app/classification_analysis.html', context)


@login_required
def classification_predict(request):
    """Page de prédiction de plateforme"""
    context = {}
    
    try:
        from .classification_module import PlatformClassificationModel
        
        # Initialiser le modèle pour charger les options
        classifier = PlatformClassificationModel()
        classifier.load_data()
        df_sample = classifier.df.copy()
        
        # Extraire les valeurs uniques pour les listes déroulantes
        if 'job_title' in df_sample.columns:
            job_titles = sorted(df_sample['job_title'].dropna().unique().tolist())
            context['job_titles'] = job_titles[:500]  # Limiter pour les performances
        
        if 'job_country' in df_sample.columns:
            countries = sorted(df_sample['job_country'].dropna().unique().tolist())
            context['countries'] = countries
        
        if 'company_name' in df_sample.columns:
            companies = sorted(df_sample['company_name'].dropna().unique().tolist())
            context['companies'] = companies[:1000]  # Limiter pour les performances
        
        # Options binaires
        context['binary_options'] = [
            {'value': 1, 'label': 'Oui'},
            {'value': 0, 'label': 'Non'}
        ]
        
        # Options de modèles
        context['models'] = ['XGBoost', 'Random Forest', 'KNN', 'SVM', 'Decision Tree']
        
    except Exception as e:
        messages.error(request, f"Erreur lors du chargement des données: {str(e)}")
    
    if request.method == 'POST':
        try:
            from .classification_module import PlatformClassificationModel
            from .models import PlatformPrediction
            
            # Récupérer les données du formulaire
            job_title = request.POST.get('job_title', 'Unknown')
            model_name = request.POST.get('model_name', None)
            
            # Initialiser et préparer le modèle
            classifier = PlatformClassificationModel()
            classifier.load_data(sample_size=50000)
            classifier.preprocess_data()
            classifier.prepare_features()
            classifier.split_data()
            
            # Charger ou entraîner les modèles
            models_loaded = classifier.load_models()
            if not models_loaded:
                classifier.train_models()
            
            # Simplifier le titre
            job_title_simplified = classifier._simplify_job_title(job_title)
            
            # Préparer les données pour la prédiction avec toutes les colonnes attendues par le modèle
            job_data = {
                'job_schedule_type': 'Full-time',  # Valeur par défaut
                'job_work_from_home': int(request.POST.get('job_work_from_home', 0)),
                'job_no_degree_mention': int(request.POST.get('job_no_degree_mention', 0)),
                'job_health_insurance': int(request.POST.get('job_health_insurance', 0)),
                'job_country': request.POST.get('job_country', 'Unknown'),
                'job_skills': 'python',  # Valeur par défaut
            }
            
            # Récupérer les données non utilisées pour l'affichage et la sauvegarde
            company_name = request.POST.get('company_name', 'Unknown')
            
            # Faire la prédiction
            platform, platform_probas = classifier.predict_platform(job_data, model_name)
            
            # Récupérer la confidence (top 1)
            confidence = platform_probas[platform]
            
            # Sauvegarder dans la base de données
            PlatformPrediction.objects.create(
                user=request.user,
                job_title_simplified=job_title_simplified,
                job_country=job_data['job_country'],
                company_name=company_name,
                job_work_from_home=bool(job_data['job_work_from_home']),
                job_no_degree_mention=bool(job_data['job_no_degree_mention']),
                job_health_insurance=bool(job_data['job_health_insurance']),
                predicted_platform=platform,
                confidence=confidence,
                model_used=model_name or classifier.best_model_name
            )
            
            messages.success(request, f"Prédiction réussie! Plateforme recommandée: {platform}")
            
            # Ajouter company_name pour l'affichage
            job_data['company_name'] = company_name
            
            context.update({
                'predicted_platform': platform,
                'platform_probas': platform_probas,
                'job_title_original': job_title,
                'job_title_simplified': job_title_simplified,
                'job_data': job_data,
                'model_used': model_name or classifier.best_model_name
            })
            
            return render(request, 'ml_app/classification_predict.html', context)
            
        except Exception as e:
            messages.error(request, f"Erreur lors de la prédiction: {str(e)}")
    
    return render(request, 'ml_app/classification_predict.html', context)


@login_required
def history(request):
    """Page d'historique des prédictions"""
    # Si l'utilisateur est admin, afficher toutes les prédictions
    if request.user.is_staff or request.user.is_superuser:
        salary_predictions = SalaryPrediction.objects.all()[:20]
        job_predictions = JobClusterPrediction.objects.all()[:20]
        clustering_results = ClusteringResult.objects.all()[:10]
        platform_predictions = PlatformPrediction.objects.all()[:20]
        job_title_predictions = JobTitlePrediction.objects.all()[:20]
        forecast_predictions = TimeSeriesPrediction.objects.all()[:20]
        experience_predictions = ExperiencePrediction.objects.all()[:20]
        employee_cluster_predictions = EmployeeClusterPrediction.objects.all()[:20]
    else:
        # Sinon, afficher seulement les prédictions de l'utilisateur
        salary_predictions = SalaryPrediction.objects.filter(user=request.user)[:20]
        job_predictions = JobClusterPrediction.objects.filter(user=request.user)[:20]
        clustering_results = ClusteringResult.objects.all()[:10]  # Les résultats de clustering sont globaux
        platform_predictions = PlatformPrediction.objects.filter(user=request.user)[:20]
        job_title_predictions = JobTitlePrediction.objects.filter(user=request.user)[:20]
        forecast_predictions = TimeSeriesPrediction.objects.filter(user=request.user)[:20]
        experience_predictions = ExperiencePrediction.objects.filter(user=request.user)[:20]
        employee_cluster_predictions = EmployeeClusterPrediction.objects.filter(user=request.user)[:20]
    
    context = {
        'salary_predictions': salary_predictions,
        'job_predictions': job_predictions,
        'clustering_results': clustering_results,
        'platform_predictions': platform_predictions,
        'job_title_predictions': job_title_predictions,
        'forecast_predictions': forecast_predictions,
        'experience_predictions': experience_predictions,
        'employee_cluster_predictions': employee_cluster_predictions
    }
    
    return render(request, 'ml_app/history.html', context)


@csrf_exempt
def chatbot_message(request):
    """API endpoint pour le chatbot"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '')
            
            if not message:
                return JsonResponse({
                    'success': False,
                    'error': 'No message provided'
                }, status=400)
            
            # Use the hardcoded API key
            api_key = "AIzaSyASA62pPJt-fF2mjvnDSvK-9-BhVMRwF5Q"
            
            # Get chatbot instance and initialize if needed
            chatbot = get_chatbot_instance(api_key=api_key)
            
            # Get answer
            result = chatbot.answer_question(message)
            
            return JsonResponse({
                'success': result['success'],
                'answer': result['answer'],
                'timestamp': data.get('timestamp', '')
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'Only POST requests are allowed'
    }, status=405)


@user_passes_test(is_admin, login_url='/login/')
def job_title_analysis(request):
    """Page d'analyse de classification des types de postes"""
    context = {}
    
    try:
        # Initialiser le modèle de classification
        job_title_model = JobTitleClassificationModel()
        job_title_model.load_data()
        job_title_model.preprocess_data()
        job_title_model.split_data()
        job_title_model.apply_smote()
        job_title_model.train_models()
        
        # Générer les visualisations
        comparison_plot = job_title_model.visualize_comparison()
        confusion_matrix_plot = job_title_model.visualize_confusion_matrix()
        class_distribution_plot = job_title_model.visualize_class_distribution()
        knn_k_values_plot = job_title_model.visualize_knn_k_values()
        roc_curves_plot = job_title_model.visualize_roc_curves()
        decision_tree_plot = job_title_model.visualize_decision_tree()
        
        # Obtenir les résultats d'évaluation
        results = job_title_model.evaluate_models()
        
        # Obtenir le rapport de classification
        classification_report = job_title_model.get_classification_report()
        
        context = {
            'comparison_plot': comparison_plot,
            'confusion_matrix_plot': confusion_matrix_plot,
            'class_distribution_plot': class_distribution_plot,
            'knn_k_values_plot': knn_k_values_plot,
            'roc_curves_plot': roc_curves_plot,
            'decision_tree_plot': decision_tree_plot,
            'results': results,
            'classification_report': classification_report,
            'best_model': job_title_model.best_model_name,
            'data_info': {
                'total_jobs': len(job_title_model.df),
                'features_used': job_title_model.features_to_use,
                'n_classes': len(job_title_model.le_target.classes_),
                'classes': job_title_model.le_target.classes_.tolist()
            }
        }
        
    except Exception as e:
        messages.error(request, f"Erreur lors de l'analyse de classification: {str(e)}")
        context['error'] = str(e)
    
    return render(request, 'ml_app/job_title_analysis.html', context)


@login_required
def job_title_predict(request):
    """Page de prédiction du type de poste"""
    # Valeurs par défaut pour les listes déroulantes
    default_schedule_types = ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship']
    default_sectors = ['Information Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 
                       'Retail', 'Telecommunications', 'Consulting', 'Government', 'Other']
    default_job_vias = ['LinkedIn', 'Indeed', 'Glassdoor', 'Monster', 'CareerBuilder', 'ZipRecruiter', 'Other']
    
    # Charger les options pour les listes déroulantes
    try:
        job_title_model = JobTitleClassificationModel()
        job_title_model.load_data()
        
        # Extraire les valeurs uniques pour les listes déroulantes
        schedule_types = sorted(job_title_model.df['job_schedule_type'].unique().tolist()) if 'job_schedule_type' in job_title_model.df.columns else default_schedule_types
        sectors = sorted(job_title_model.df['sector'].unique().tolist()) if 'sector' in job_title_model.df.columns else default_sectors
        job_vias = sorted(job_title_model.df['job_via'].unique().tolist()) if 'job_via' in job_title_model.df.columns else default_job_vias
        
        context = {
            'schedule_types': schedule_types,
            'sectors': sectors,
            'job_vias': job_vias,
            'skills_range': range(0, 6)  # 0 à 5 compétences
        }
    except Exception as e:
        messages.warning(request, f"Utilisation des valeurs par défaut. Dataset introuvable.")
        context = {
            'schedule_types': default_schedule_types,
            'sectors': default_sectors,
            'job_vias': default_job_vias,
            'skills_range': range(0, 6)
        }
    
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            job_data = {
                'job_schedule_type': request.POST.get('job_schedule_type'),
                'sector': request.POST.get('sector'),
                'job_via': request.POST.get('job_via'),
                'job_skills': int(request.POST.get('job_skills'))
            }
            
            # Initialiser et préparer le modèle
            job_title_model = JobTitleClassificationModel()
            job_title_model.load_data()
            job_title_model.preprocess_data()
            job_title_model.split_data()
            job_title_model.train_models()
            
            # Faire la prédiction
            result = job_title_model.predict_job_title(job_data)
            
            if isinstance(result, tuple):
                predicted_job_title, confidence = result
            else:
                predicted_job_title = result
                confidence = None
            
            # Sauvegarder dans la base de données
            JobTitlePrediction.objects.create(
                user=request.user,
                job_schedule_type=job_data['job_schedule_type'],
                sector=job_data['sector'],
                job_via=job_data['job_via'],
                job_skills=job_data['job_skills'],
                predicted_job_title=predicted_job_title,
                confidence=confidence * 100 if confidence else None,
                model_used=job_title_model.best_model_name
            )
            
            messages.success(request, f"Prédiction réussie! Type de poste prédit: {predicted_job_title}")
            
            context.update({
                'predicted_job_title': predicted_job_title,
                'confidence': confidence * 100 if confidence else None,
                'model_used': job_title_model.best_model_name,
                'job_data': job_data
            })
            
            return render(request, 'ml_app/job_title_predict.html', context)
            
        except Exception as e:
            messages.error(request, f"Erreur lors de la prédiction: {str(e)}")
    
    return render(request, 'ml_app/job_title_predict.html', context)


@user_passes_test(is_admin, login_url='/login/')
def forecast_analysis(request):
    """Page d'analyse des prévisions de séries temporelles"""
    context = {}
    
    try:
        # Initialiser le modèle
        forecast_model = TimeSeriesForecastModel()
        
        # Charger et préparer les données
        forecast_model.load_data()
        forecast_model.prepare_time_series()
        
        # Décomposer la série
        forecast_model.decompose_series()
        
        # Obtenir les statistiques
        stats = forecast_model.get_statistics()
        
        # Comparer les modèles
        model_comparison = forecast_model.compare_models()
        
        # Générer les visualisations
        plot_time_series = forecast_model.visualize_time_series()
        plot_decomposition = forecast_model.visualize_decomposition()
        plot_comparison = forecast_model.visualize_model_comparison()
        plot_predictions = forecast_model.visualize_predictions(3)
        
        context = {
            'stats': stats,
            'model_comparison': model_comparison,
            'plot_time_series': plot_time_series,
            'plot_decomposition': plot_decomposition,
            'plot_comparison': plot_comparison,
            'plot_predictions': plot_predictions,
            'best_model': forecast_model.best_model_name
        }
        
    except Exception as e:
        messages.error(request, f"Erreur lors de l'analyse: {str(e)}")
    
    return render(request, 'ml_app/forecast_analysis.html', context)


@login_required
def forecast_predict(request):
    """Page de prédiction de séries temporelles"""
    context = {}
    
    if request.method == 'POST':
        try:
            num_months = int(request.POST.get('num_months', 3))
            
            if num_months < 1 or num_months > 12:
                messages.error(request, "Le nombre de mois doit être entre 1 et 12")
                return render(request, 'ml_app/forecast_predict.html', context)
            
            # Initialiser le modèle
            forecast_model = TimeSeriesForecastModel()
            forecast_model.load_data()
            forecast_model.prepare_time_series()
            forecast_model.decompose_series()
            
            # Faire la prédiction
            predictions_data = forecast_model.predict_next_months(num_months)
            
            # Générer la visualisation
            plot_predictions = forecast_model.visualize_predictions(num_months)
            
            # Zip dates and predictions together for template iteration
            predictions_list = [
                {'date': date, 'value': pred}
                for date, pred in zip(predictions_data['dates'], predictions_data['predictions'])
            ]
            
            # Sauvegarder dans la base de données
            TimeSeriesPrediction.objects.create(
                user=request.user,
                num_months=num_months,
                predictions=predictions_data,
                model_used=predictions_data['model'],
                sma_value=predictions_data['sma_6']
            )
            
            messages.success(request, f"Prédiction réussie pour {num_months} mois!")
            
            context.update({
                'predictions_data': predictions_data,
                'predictions_list': predictions_list,
                'plot_predictions': plot_predictions,
                'num_months': num_months
            })
            
            return render(request, 'ml_app/forecast_predict.html', context)
            
        except Exception as e:
            messages.error(request, f"Erreur lors de la prédiction: {str(e)}")
    
    return render(request, 'ml_app/forecast_predict.html', context)


# ========================================
# EXPERIENCE PREDICTION VIEWS
# ========================================

@user_passes_test(is_admin)
def experience_analysis(request):
    """
    Page d'analyse pour la prédiction des années d'expérience (Admin seulement)
    """
    try:
        # Charger le modèle
        exp_model = ExperiencePredictionModel()
        
        # Entraîner tous les modèles
        exp_model.train_all_models()
        
        # Évaluer les modèles
        exp_model.evaluate_models()
        
        # Obtenir le meilleur modèle
        best_model_name, best_metrics = exp_model.get_best_model()
        
        # Générer les visualisations
        plot_comparison = exp_model.visualize_model_comparison()
        plot_actual_vs_pred = exp_model.visualize_actual_vs_predicted(best_model_name)
        plot_residuals = exp_model.visualize_residuals(best_model_name)
        plot_feature_importance = exp_model.visualize_feature_importance()
        
        # Statistiques globales
        stats = {
            'total_samples': len(exp_model.df),
            'train_samples': len(exp_model.X_train),
            'test_samples': len(exp_model.X_test),
            'avg_years_experience': round(exp_model.df[exp_model.target_column].mean(), 2),
            'min_years_experience': round(exp_model.df[exp_model.target_column].min(), 2),
            'max_years_experience': round(exp_model.df[exp_model.target_column].max(), 2)
        }
        
        context = {
            'stats': stats,
            'best_model': best_model_name,
            'best_metrics': best_metrics,
            'all_metrics': exp_model.metrics,
            'plot_comparison': plot_comparison,
            'plot_actual_vs_pred': plot_actual_vs_pred,
            'plot_residuals': plot_residuals,
            'plot_feature_importance': plot_feature_importance,
        }
        
        return render(request, 'ml_app/experience_analysis.html', context)
        
    except Exception as e:
        messages.error(request, f"Erreur lors de l'analyse: {str(e)}")
        return redirect('ml_app:home')


@login_required
def experience_predict(request):
    """
    Page de prédiction des années d'expérience
    """
    context = {}
    
    # Charger le modèle
    try:
        exp_model = ExperiencePredictionModel()
        exp_model.train_all_models()
        
        # Obtenir les valeurs uniques pour les dropdowns
        context['industries'] = sorted(exp_model.df['industry'].unique().tolist())
        context['education_levels'] = sorted(exp_model.df['education_required'].unique().tolist())
        context['skills'] = sorted(exp_model.df['required_skills'].unique().tolist())
        context['employment_types'] = sorted(exp_model.df['employment_type'].unique().tolist())
        context['experience_levels'] = sorted(exp_model.df['experience_level'].unique().tolist())
        context['company_sizes'] = sorted(exp_model.df['company_size'].unique().tolist())
        
    except Exception as e:
        messages.error(request, f"Erreur lors du chargement du modèle: {str(e)}")
        return redirect('ml_app:home')
    
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            input_data = {
                'industry': request.POST.get('industry'),
                'education_required': request.POST.get('education_required'),
                'required_skills': request.POST.get('required_skills'),
                'employment_type': request.POST.get('employment_type'),
                'experience_level': request.POST.get('experience_level'),
                'company_size': request.POST.get('company_size')
            }
            
            # Faire la prédiction
            result = exp_model.predict_experience(input_data)
            
            # Sauvegarder dans la base de données
            ExperiencePrediction.objects.create(
                user=request.user,
                industry=input_data['industry'],
                education_required=input_data['education_required'],
                required_skills=input_data['required_skills'],
                employment_type=input_data['employment_type'],
                experience_level=input_data['experience_level'],
                company_size=input_data['company_size'],
                predicted_years=result['best_prediction'],
                model_used=result['best_model'],
                all_predictions=result['predictions']
            )
            
            messages.success(request, f"Prédiction réussie! Années d'expérience estimées: {result['best_prediction']}")
            
            context.update({
                'result': result,
                'input_data': input_data
            })
            
            return render(request, 'ml_app/experience_predict.html', context)
            
        except Exception as e:
            messages.error(request, f"Erreur lors de la prédiction: {str(e)}")
    
    return render(request, 'ml_app/experience_predict.html', context)


# ========================================
# EMPLOYEE CLUSTERING VIEWS
# ========================================

@user_passes_test(is_admin)
def employee_clustering_analysis(request):
    """
    Page d'analyse pour le clustering des employés (Admin seulement)
    """
    try:
        # Charger le modèle
        cluster_model = EmployeeClusteringModel()
        
        # Entraîner tous les modèles avec 2 clusters (Junior/Senior)
        cluster_model.train_all_models(n_clusters=2)
        
        # Évaluer les modèles
        metrics, best_model = cluster_model.evaluate_models()
        
        # Obtenir les statistiques par cluster
        cluster_stats = cluster_model.get_cluster_statistics()
        
        # Générer les visualisations
        plot_kmeans_optimization = cluster_model.visualize_kmeans_optimization()
        plot_cluster_profiles = cluster_model.visualize_cluster_profiles('Agglomerative')
        plot_boxplots = cluster_model.visualize_boxplots('Agglomerative')
        plot_named_clusters = cluster_model.visualize_named_clusters()
        
        context = {
            'metrics': metrics,
            'best_model': best_model,
            'cluster_stats': cluster_stats,
            'plot_kmeans_optimization': plot_kmeans_optimization,
            'plot_cluster_profiles': plot_cluster_profiles,
            'plot_boxplots': plot_boxplots,
            'plot_named_clusters': plot_named_clusters,
            'total_employees': len(cluster_model.df)
        }
        
        return render(request, 'ml_app/employee_clustering_analysis.html', context)
        
    except Exception as e:
        messages.error(request, f"Erreur lors de l'analyse: {str(e)}")
        return redirect('ml_app:home')


@login_required
def employee_clustering_predict(request):
    """
    Page de prédiction de cluster d'employé (Junior/Senior)
    """
    context = {}
    
    # Charger le modèle
    try:
        cluster_model = EmployeeClusteringModel()
        cluster_model.train_all_models(n_clusters=2)
        
        # Obtenir les statistiques pour afficher des valeurs de référence
        cluster_stats = cluster_model.get_cluster_statistics()
        context['cluster_stats'] = cluster_stats
        
    except Exception as e:
        messages.error(request, f"Erreur lors du chargement du modèle: {str(e)}")
        return redirect('ml_app:home')
    
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            input_data = {
                'salary_usd': float(request.POST.get('salary_usd')),
                'benefits_score': float(request.POST.get('benefits_score')),
                'years_experience': float(request.POST.get('years_experience')),
                'remote_ratio': float(request.POST.get('remote_ratio'))
            }
            
            # Faire la prédiction
            result = cluster_model.predict_cluster(input_data)
            
            # Sauvegarder dans la base de données
            EmployeeClusterPrediction.objects.create(
                user=request.user,
                salary_usd=input_data['salary_usd'],
                benefits_score=input_data['benefits_score'],
                years_experience=input_data['years_experience'],
                remote_ratio=input_data['remote_ratio'],
                predicted_cluster=result['cluster_name'],
                kmeans_cluster=result['kmeans_cluster'],
                agglomerative_cluster=result['agglomerative_cluster']
            )
            
            messages.success(request, f"Prédiction réussie! Profil: {result['cluster_name']}")
            
            context.update({
                'result': result,
                'input_data': input_data
            })
            
            return render(request, 'ml_app/employee_clustering_predict.html', context)
            
        except Exception as e:
            messages.error(request, f"Erreur lors de la prédiction: {str(e)}")
    
    return render(request, 'ml_app/employee_clustering_predict.html', context)


# ========================================
# DS1: Company Profiling & Segmentation
# ========================================

def is_admin(user):
    """Vérifie si l'utilisateur est un administrateur"""
    return user.is_authenticated and user.is_superuser


@user_passes_test(is_admin, login_url='ml_app:login')
def company_profiling_analysis(request):
    """Analyse de segmentation des entreprises (Admin uniquement)"""
    try:
        model = CompanyProfilingModel()
        
        # Charger et préparer les données
        model.load_data()
        model.prepare_features()
        
        # Entraîner K-Means avec K=4
        training_result = model.train_kmeans(n_clusters=4)
        
        # Obtenir les profils des clusters
        cluster_profiles = model.get_cluster_profiles()
        
        # Générer les visualisations
        plot_elbow = model.plot_elbow_method()
        plot_pca = model.plot_pca_clusters()
        plot_composition = model.plot_cluster_composition()
        plot_profiling = model.plot_profiling_bars()
        
        context = {
            'training_result': training_result,
            'cluster_profiles': cluster_profiles.to_dict('records') if cluster_profiles is not None else [],
            'plot_elbow': plot_elbow,
            'plot_pca': plot_pca,
            'plot_composition': plot_composition,
            'plot_profiling': plot_profiling
        }
        
        return render(request, 'ml_app/company_profiling_analysis.html', context)
    
    except Exception as e:
        messages.error(request, f"Erreur lors de l'analyse: {str(e)}")
        return render(request, 'ml_app/company_profiling_analysis.html', {})


@login_required(login_url='ml_app:login')
def company_profiling_predict(request):
    """Prédiction de segmentation pour une entreprise"""
    # Options pour les formulaires
    context = {
        'sectors': ['Technology', 'Healthcare', 'Finance', 'Education', 'Retail', 
                   'Manufacturing', 'Consulting', 'Media', 'Energy', 'Automotive'],
        'company_sizes': ['S', 'M', 'L'],
        'ai_profiles': ['High AI', 'Low AI']
    }
    
    if request.method == 'POST':
        try:
            # Récupérer les inputs
            input_data = {
                'sector_group': request.POST.get('sector_group'),
                'company_size': request.POST.get('company_size'),
                'ai_profile': request.POST.get('ai_profile')
            }
            
            # Charger et préparer le modèle
            model = CompanyProfilingModel()
            
            if not model.load_data():
                messages.error(request, "Erreur lors du chargement des données")
                return render(request, 'ml_app/company_profiling_predict.html', context)
            
            if not model.prepare_features():
                messages.error(request, "Erreur lors de la préparation des features")
                return render(request, 'ml_app/company_profiling_predict.html', context)
            
            training_result = model.train_kmeans(n_clusters=4)
            if training_result is None:
                messages.error(request, "Erreur lors de l'entraînement du modèle")
                return render(request, 'ml_app/company_profiling_predict.html', context)
            
            # Prédire le cluster
            predicted_cluster = model.predict_cluster(
                sector_group=input_data['sector_group'],
                company_size=input_data['company_size'],
                ai_profile=input_data['ai_profile']
            )
            
            if predicted_cluster is not None:
                # Sauvegarder dans la DB
                CompanySegmentation.objects.create(
                    user=request.user,
                    sector_group=input_data['sector_group'],
                    company_size=input_data['company_size'],
                    ai_profile=input_data['ai_profile'],
                    predicted_cluster=predicted_cluster
                )
                
                # Obtenir le profil du cluster
                cluster_profiles = model.get_cluster_profiles()
                cluster_info = {}
                if cluster_profiles is not None and not cluster_profiles.empty:
                    cluster_data = cluster_profiles[cluster_profiles['cluster_id'] == predicted_cluster]
                    if not cluster_data.empty:
                        cluster_info = cluster_data.to_dict('records')[0]
                
                messages.success(request, f"Segmentation réussie! Cluster: {predicted_cluster}")
                
                context.update({
                    'result': {
                        'cluster_id': predicted_cluster,
                        'cluster_info': cluster_info
                    },
                    'input_data': input_data
                })
            else:
                messages.error(request, "Erreur lors de la prédiction")
                
        except Exception as e:
            messages.error(request, f"Erreur: {str(e)}")
    
    return render(request, 'ml_app/company_profiling_predict.html', context)


# ========================================
# DS2: Job Analysis (Regression & Classification)
# ========================================

@user_passes_test(is_admin, login_url='ml_app:login')
def job_analysis_analysis(request):
    """Analyse des emplois - régression et classification (Admin uniquement)"""
    try:
        model = JobAnalysisModel()
        
        # Charger et préparer les données
        model.load_data()
        model.prepare_features()
        
        # Entraîner les modèles
        regression_metrics = model.train_regression_models()
        classification_metrics = model.train_classification_model()
        
        # Générer les visualisations
        plot_regression = model.plot_regression_comparison()
        plot_importance = model.plot_feature_importance()
        plot_confusion = model.plot_confusion_matrix()
        plot_classification = model.plot_classification_report()
        
        context = {
            'regression_metrics': regression_metrics,
            'classification_metrics': {
                'accuracy': classification_metrics['accuracy'],
                'confusion_matrix': classification_metrics['confusion_matrix'].tolist()
            },
            'plot_regression': plot_regression,
            'plot_importance': plot_importance,
            'plot_confusion': plot_confusion,
            'plot_classification': plot_classification
        }
        
        return render(request, 'ml_app/job_analysis_analysis.html', context)
    
    except Exception as e:
        messages.error(request, f"Erreur lors de l'analyse: {str(e)}")
        return render(request, 'ml_app/job_analysis_analysis.html', {})


@login_required(login_url='ml_app:login')
def job_analysis_predict_salary(request):
    """Prédiction du salaire"""
    context = {
        'company_sizes': ['S', 'M', 'L'],
        'seniorities': ['Junior', 'Senior'],
        'ai_profiles': ['High AI', 'Low AI'],
        'sectors': ['Technology', 'Healthcare', 'Finance', 'Education', 'Retail', 
                   'Manufacturing', 'Consulting', 'Media', 'Energy', 'Automotive'],
        'countries': ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 
                     'India', 'China', 'Japan', 'Australia', 'Brazil']
    }
    
    if request.method == 'POST':
        try:
            # Récupérer les inputs
            input_data = {
                'year_experience': float(request.POST.get('year_experience')),
                'nb_skills': int(request.POST.get('nb_skills')),
                'job_description_length': int(request.POST.get('job_description_length')),
                'benefits_score': float(request.POST.get('benefits_score')),
                'company_size': request.POST.get('company_size'),
                'seniority': request.POST.get('seniority'),
                'ai_profile': request.POST.get('ai_profile'),
                'sector_group': request.POST.get('sector_group'),
                'company_location': request.POST.get('company_location')
            }
            
            model_type = request.POST.get('model_type', 'random_forest')
            
            # Charger et préparer le modèle
            model = JobAnalysisModel()
            
            if not model.load_data():
                messages.error(request, "Erreur lors du chargement des données")
                return render(request, 'ml_app/job_analysis_predict_salary.html', context)
            
            if not model.prepare_features():
                messages.error(request, "Erreur lors de la préparation des features")
                return render(request, 'ml_app/job_analysis_predict_salary.html', context)
            
            regression_result = model.train_regression_models()
            if regression_result is None:
                messages.error(request, "Erreur lors de l'entraînement du modèle")
                return render(request, 'ml_app/job_analysis_predict_salary.html', context)
            
            # Prédire
            result = model.predict_salary(**input_data, model_type=model_type)
            
            if result:
                # Sauvegarder
                JobAnalysisPrediction.objects.create(
                    user=request.user,
                    year_experience=input_data['year_experience'],
                    nb_skills=input_data['nb_skills'],
                    job_description_length=input_data['job_description_length'],
                    benefits_score=input_data['benefits_score'],
                    company_size=input_data['company_size'],
                    seniority=input_data['seniority'],
                    ai_profile=input_data['ai_profile'],
                    sector_group=input_data['sector_group'],
                    company_location=input_data['company_location'],
                    predicted_salary_usd=result['salary_usd'],
                    salary_model_used=result['model_used'],
                    prediction_type='salary'
                )
                
                messages.success(request, f"Prédiction réussie! Salaire estimé: ${result['salary_usd']:.2f}")
                
                context.update({
                    'result': result,
                    'input_data': input_data
                })
            else:
                messages.error(request, "Erreur lors de la prédiction")
                
        except Exception as e:
            messages.error(request, f"Erreur: {str(e)}")
    
    return render(request, 'ml_app/job_analysis_predict_salary.html', context)


@login_required(login_url='ml_app:login')
def job_analysis_predict_ai_profile(request):
    """Prédiction du profil IA"""
    context = {
        'company_sizes': ['S', 'M', 'L'],
        'seniorities': ['Junior', 'Senior'],
        'sectors': ['Technology', 'Healthcare', 'Finance', 'Education', 'Retail', 
                   'Manufacturing', 'Consulting', 'Media', 'Energy', 'Automotive'],
        'countries': ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 
                     'India', 'China', 'Japan', 'Australia', 'Brazil']
    }
    
    if request.method == 'POST':
        try:
            # Récupérer les inputs
            input_data = {
                'salary_usd': float(request.POST.get('salary_usd')),
                'year_experience': float(request.POST.get('year_experience')),
                'nb_skills': int(request.POST.get('nb_skills')),
                'job_description_length': int(request.POST.get('job_description_length')),
                'benefits_score': float(request.POST.get('benefits_score')),
                'company_size': request.POST.get('company_size'),
                'seniority': request.POST.get('seniority'),
                'sector_group': request.POST.get('sector_group'),
                'company_location': request.POST.get('company_location')
            }
            
            # Charger et préparer le modèle
            model = JobAnalysisModel()
            
            if not model.load_data():
                messages.error(request, "Erreur lors du chargement des données")
                return render(request, 'ml_app/job_analysis_predict_ai_profile.html', context)
            
            if not model.prepare_features():
                messages.error(request, "Erreur lors de la préparation des features")
                return render(request, 'ml_app/job_analysis_predict_ai_profile.html', context)
            
            classification_result = model.train_classification_model()
            if classification_result is None:
                messages.error(request, "Erreur lors de l'entraînement du modèle")
                return render(request, 'ml_app/job_analysis_predict_ai_profile.html', context)
            
            # Prédire
            result = model.predict_ai_profile(**input_data)
            
            if result:
                # Sauvegarder
                JobAnalysisPrediction.objects.create(
                    user=request.user,
                    year_experience=input_data['year_experience'],
                    nb_skills=input_data['nb_skills'],
                    job_description_length=input_data['job_description_length'],
                    benefits_score=input_data['benefits_score'],
                    company_size=input_data['company_size'],
                    seniority=input_data['seniority'],
                    ai_profile='Unknown',  # Placeholder car c'est ce qu'on prédit
                    sector_group=input_data['sector_group'],
                    company_location=input_data['company_location'],
                    predicted_ai_profile=result['ai_profile'],
                    ai_profile_confidence=result['confidence'],
                    prediction_type='ai_profile'
                )
                
                messages.success(request, f"Prédiction réussie! Profil: {result['ai_profile']} ({result['confidence']*100:.1f}%)")
                
                context.update({
                    'result': result,
                    'input_data': input_data
                })
            else:
                messages.error(request, "Erreur lors de la prédiction")
                
        except Exception as e:
            messages.error(request, f"Erreur: {str(e)}")
    
    return render(request, 'ml_app/job_analysis_predict_ai_profile.html', context)


def is_admin(user):
    """Vérifie si l'utilisateur est admin"""
    return user.is_staff


@login_required
@user_passes_test(is_admin)
def powerbi_dashboard(request):
    """Affiche le dashboard Power BI - réservé aux admins"""
    from django.conf import settings
    
    # Récupérer l'URL depuis les settings
    powerbi_url = getattr(settings, 'POWERBI_EMBED_URL', None)
    
    context = {
        'powerbi_embed_url': powerbi_url
    }
    
    return render(request, 'ml_app/powerbi_dashboard.html', context)


