from django.db import models
from django.contrib.auth.models import User

class ClusteringResult(models.Model):
    """Stocke les résultats du clustering"""
    created_at = models.DateTimeField(auto_now_add=True)
    n_clusters = models.IntegerField()
    algorithm = models.CharField(max_length=50)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.algorithm} - {self.n_clusters} clusters - {self.created_at}"


class SalaryPrediction(models.Model):
    """Stocke les prédictions de salaire"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    age = models.FloatField()
    years_of_experience = models.FloatField()
    gender = models.CharField(max_length=50)
    education_level = models.CharField(max_length=100)
    job_title = models.CharField(max_length=200)
    predicted_salary = models.FloatField()
    model_used = models.CharField(max_length=100)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Prediction: {self.predicted_salary} - {self.created_at}"


class JobClusterPrediction(models.Model):
    """Stocke les prédictions de cluster pour les jobs AI"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    job_title = models.CharField(max_length=200)
    salary_usd = models.FloatField()
    experience_level = models.CharField(max_length=50)
    employment_type = models.CharField(max_length=50)
    company_location = models.CharField(max_length=100)
    company_size = models.CharField(max_length=50)
    years_experience = models.FloatField()
    predicted_cluster = models.IntegerField()
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Job Cluster: {self.predicted_cluster} - {self.created_at}"


class PlatformPrediction(models.Model):
    """Stocke les prédictions de plateformes de recrutement"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    job_title_simplified = models.CharField(max_length=200)
    job_country = models.CharField(max_length=100)
    company_name = models.CharField(max_length=200)
    job_work_from_home = models.BooleanField()
    job_no_degree_mention = models.BooleanField()
    job_health_insurance = models.BooleanField()
    predicted_platform = models.CharField(max_length=100)
    confidence = models.FloatField()
    model_used = models.CharField(max_length=100)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Platform: {self.predicted_platform} ({self.confidence}%) - {self.created_at}"


class JobTitlePrediction(models.Model):
    """Stocke les prédictions de type de poste (Job Title Classification)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    job_schedule_type = models.CharField(max_length=100)
    sector = models.CharField(max_length=200)
    job_via = models.CharField(max_length=100)
    job_skills = models.IntegerField()
    predicted_job_title = models.CharField(max_length=200)
    confidence = models.FloatField(null=True, blank=True)
    model_used = models.CharField(max_length=100)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Job Title: {self.predicted_job_title} ({self.confidence}%) - {self.created_at}"


class TimeSeriesPrediction(models.Model):
    """Stocke les prédictions de séries temporelles (prévisions mensuelles d'offres)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    num_months = models.IntegerField()  # Nombre de mois prédits
    predictions = models.JSONField()  # Liste des prédictions
    model_used = models.CharField(max_length=100)
    sma_value = models.FloatField()  # Valeur SMA utilisée
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Forecast: {self.num_months} mois - {self.created_at}"


class ExperiencePrediction(models.Model):
    """Stocke les prédictions des années d'expérience"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    industry = models.CharField(max_length=200)
    education_required = models.CharField(max_length=100)
    required_skills = models.CharField(max_length=500)
    employment_type = models.CharField(max_length=50)
    experience_level = models.CharField(max_length=50)
    company_size = models.CharField(max_length=50)
    predicted_years = models.FloatField()
    model_used = models.CharField(max_length=100)
    all_predictions = models.JSONField()  # Prédictions de tous les modèles
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Experience: {self.predicted_years} ans - {self.created_at}"


class EmployeeClusterPrediction(models.Model):
    """Stocke les prédictions de clustering d'employés (Junior/Senior)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    salary_usd = models.FloatField()
    benefits_score = models.FloatField()
    years_experience = models.FloatField()
    remote_ratio = models.FloatField()
    predicted_cluster = models.CharField(max_length=50)  # Junior ou Senior
    kmeans_cluster = models.IntegerField(null=True, blank=True)
    agglomerative_cluster = models.IntegerField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Cluster: {self.predicted_cluster} - {self.created_at}"


class CompanySegmentation(models.Model):
    """Stocke les résultats de segmentation d'entreprises (DS1)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    sector_group = models.CharField(max_length=200)
    company_size = models.CharField(max_length=50)
    ai_profile = models.CharField(max_length=50)
    predicted_cluster = models.IntegerField()
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Segment: Cluster {self.predicted_cluster} - {self.created_at}"


class JobAnalysisPrediction(models.Model):
    """Stocke les prédictions d'analyse d'emplois (DS2 - Régression et Classification)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Features d'entrée
    year_experience = models.FloatField()
    nb_skills = models.IntegerField()
    job_description_length = models.IntegerField()
    benefits_score = models.FloatField()
    company_size = models.CharField(max_length=50)
    seniority = models.CharField(max_length=50)
    ai_profile = models.CharField(max_length=50)
    sector_group = models.CharField(max_length=200)
    company_location = models.CharField(max_length=100)
    
    # Prédiction de salaire (régression)
    predicted_salary_usd = models.FloatField(null=True, blank=True)
    salary_model_used = models.CharField(max_length=100, null=True, blank=True)
    
    # Prédiction du profil IA (classification)
    predicted_ai_profile = models.CharField(max_length=50, null=True, blank=True)
    ai_profile_confidence = models.FloatField(null=True, blank=True)
    
    # Type de prédiction effectuée
    prediction_type = models.CharField(max_length=50)  # 'salary', 'ai_profile', ou 'both'
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Job Analysis: {self.prediction_type} - {self.created_at}"
