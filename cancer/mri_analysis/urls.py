from django.urls import path
from . import views

urlpatterns = [
    path('', views.analyze_mri, name='analyze_mri'),  # Upload MRI image
    path('result/<int:pk>/', views.result_view, name='result_view'),  # View results
]
