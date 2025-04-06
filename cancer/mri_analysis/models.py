from django.db import models
from django.utils import timezone

class MRIImage(models.Model):
    image = models.ImageField(upload_to='mri_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class AnalysisResult(models.Model):
    mri_image = models.ForeignKey(MRIImage, on_delete=models.CASCADE)
    tumor_detected = models.BooleanField()
    analyzed_at = models.DateTimeField(default=timezone.now)

    detailed_analysis = models.TextField(blank=True, null=True)
    
    segmented_result = models.ImageField(upload_to='tumor_mask/', blank=True, null=True)
    bounded_box_image = models.ImageField(upload_to='with_bounding_box/', blank=True, null=True)

    tumor_size_px = models.IntegerField(blank=True, null=True)
    tumor_size_mm2 = models.FloatField(blank=True, null=True)
