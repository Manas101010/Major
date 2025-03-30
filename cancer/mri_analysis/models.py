from django.db import models

class MRIImage(models.Model):
    image = models.ImageField(upload_to='mri_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

from django.utils import timezone

class AnalysisResult(models.Model):
    mri_image = models.ForeignKey(MRIImage, on_delete=models.CASCADE)
    tumor_detected = models.BooleanField()
    analyzed_at = models.DateTimeField(default=timezone.now)  # Manually set a default value
    detailed_analysis = models.TextField(blank=True, null=True)
    segmented_result=models.ImageField(upload_to='segmentation_results/', blank=True, null=True)
