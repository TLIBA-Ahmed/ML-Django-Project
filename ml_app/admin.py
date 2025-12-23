from django.contrib import admin
from .models import ClusteringResult, SalaryPrediction, JobClusterPrediction, PlatformPrediction

@admin.register(ClusteringResult)
class ClusteringResultAdmin(admin.ModelAdmin):
    list_display = ['algorithm', 'n_clusters', 'created_at']
    list_filter = ['algorithm', 'created_at']
    search_fields = ['algorithm']


@admin.register(SalaryPrediction)
class SalaryPredictionAdmin(admin.ModelAdmin):
    list_display = ['job_title', 'predicted_salary', 'model_used', 'created_at']
    list_filter = ['model_used', 'education_level', 'created_at']
    search_fields = ['job_title', 'gender', 'education_level']


@admin.register(JobClusterPrediction)
class JobClusterPredictionAdmin(admin.ModelAdmin):
    list_display = ['job_title', 'company_location', 'predicted_cluster', 'created_at']
    list_filter = ['predicted_cluster', 'experience_level', 'created_at']
    search_fields = ['job_title', 'company_location']


@admin.register(PlatformPrediction)
class PlatformPredictionAdmin(admin.ModelAdmin):
    list_display = ['job_title_simplified', 'predicted_platform', 'confidence', 'model_used', 'created_at']
    list_filter = ['predicted_platform', 'model_used', 'job_country', 'created_at']
    search_fields = ['job_title_simplified', 'company_name', 'predicted_platform']
