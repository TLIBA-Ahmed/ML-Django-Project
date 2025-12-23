from django.urls import path
from . import views

app_name = 'ml_app'

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    path('clustering/', views.clustering_analysis, name='clustering_analysis'),
    path('clustering/predict/', views.clustering_predict, name='clustering_predict'),
    path('salary/', views.salary_analysis, name='salary_analysis'),
    path('salary/predict/', views.salary_predict, name='salary_predict'),
    path('classification/', views.classification_analysis, name='classification_analysis'),
    path('classification/predict/', views.classification_predict, name='classification_predict'),
    path('job-title/', views.job_title_analysis, name='job_title_analysis'),
    path('job-title/predict/', views.job_title_predict, name='job_title_predict'),
    path('forecast/', views.forecast_analysis, name='forecast_analysis'),
    path('forecast/predict/', views.forecast_predict, name='forecast_predict'),
    path('experience/', views.experience_analysis, name='experience_analysis'),
    path('experience/predict/', views.experience_predict, name='experience_predict'),
    path('employee-clustering/', views.employee_clustering_analysis, name='employee_clustering_analysis'),
    path('employee-clustering/predict/', views.employee_clustering_predict, name='employee_clustering_predict'),
    path('company-profiling/', views.company_profiling_analysis, name='company_profiling_analysis'),
    path('company-profiling/predict/', views.company_profiling_predict, name='company_profiling_predict'),
    path('job-analysis/', views.job_analysis_analysis, name='job_analysis_analysis'),
    path('job-analysis/predict/salary/', views.job_analysis_predict_salary, name='job_analysis_predict_salary'),
    path('job-analysis/predict/ai-profile/', views.job_analysis_predict_ai_profile, name='job_analysis_predict_ai_profile'),
    path('powerbi-dashboard/', views.powerbi_dashboard, name='powerbi_dashboard'),
    path('history/', views.history, name='history'),
    path('chatbot/message/', views.chatbot_message, name='chatbot_message'),
]
