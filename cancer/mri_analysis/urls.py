from django.urls import path
from . import views

urlpatterns = [
    path('mri_analysis/', views.analyze_mri, name='analyze_mri'),  # Upload MRI image
    path('result/<int:pk>/', views.result_view, name='result_view'),  # View results
    path('', views.brain_score_view, name='brain_score'),
    path('predict/', views.predict_tumor_stage, name='predict_tumor_stage'),
]
