from django.urls import path
from . import views

urlpatterns = [
    path('', views.landingPage, name='landing'),  # Home page
    path('auth/', views.signin, name='auth'),  # Signin Page
    path('mri_analysis/', views.analyze_mri, name='analyze_mri'),  # Upload MRI image
    path('result/<int:pk>/', views.result_view, name='result_view'),  # View results
    path('brainscore', views.brain_score_view, name='brain_score'),
    path('predict/', views.predict_tumor_stage, name='predict_tumor_stage'),
    path('generate-pdf/', views.generate_pdf, name='generate_pdf'),
]
