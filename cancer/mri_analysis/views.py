import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image
from .models import MRIImage, AnalysisResult
from .classification import make_prediction  # Import classification logic
from .segmentation import process_uploaded_image  # Updated function call
from .fuzzy_logic import assess_brain_score
from .bs_form import BrainScoreForm
from .stage_predictor import predict_stage
from .tp_form import TumorPredictionForm

# Brain Score Assessment
def brain_score_view(request):
    result = None
    score = None

    if request.method == 'POST':
        form = BrainScoreForm(request.POST)
        if form.is_valid():
            data = list(map(int, form.cleaned_data.values()))
            score = assess_brain_score(data)
            if score >= 50:
                result = "Brain Score is Good 🙂"
            else:
                result = "Brain Score is Bad 🙁 (Recommendation: MRI Test)"
    else:
        form = BrainScoreForm()

    return render(request, 'brain_score.html', {'form': form, 'result': result, 'score': score})

def analyze_mri(request):
    if request.method == 'POST' and 'mri_image' in request.FILES:
        uploaded_file = request.FILES['mri_image']
        mri_image = MRIImage.objects.create(image=uploaded_file)

        # Step 1: Tumor Classification
        prediction_label, confidence_score = make_prediction(mri_image.image.path)
        print(f"Prediction Label: {prediction_label}, Confidence: {confidence_score}")

        tumor_detected = prediction_label.lower() != 'notumor'
        print(f"Tumor Detected: {tumor_detected}")

        # Step 2: Create initial analysis record
        analysis_result = AnalysisResult.objects.create(
            mri_image=mri_image,
            tumor_detected=tumor_detected,
            detailed_analysis=prediction_label
        )

        # Step 3: Run segmentation if tumor detected
        if tumor_detected:
            try:
                # Run segmentation and get visual results
                image_with_box_url, tumor_mask_url, tumor_pixels, tumor_size_mm2 = process_uploaded_image(mri_image.image)

                # Update analysis result
                analysis_result.segmented_result = tumor_mask_url.replace('media/', '')
                analysis_result.bounded_box_image = image_with_box_url.replace('media/', '')
                analysis_result.tumor_size_px = tumor_pixels
                analysis_result.tumor_size_mm2 = tumor_size_mm2

            except Exception as e:
                print(f"Segmentation Error: {e}")

        # Step 4: Save and redirect
        analysis_result.save()
        return redirect('result_view', pk=analysis_result.pk)

    return render(request, 'upload_mri.html')

def result_view(request, pk):
    analysis_result = get_object_or_404(AnalysisResult, pk=pk)

    print(f"Analysis Result ID: {analysis_result.pk}, Tumor Detected: {analysis_result.tumor_detected}")

    return render(request, 'result_view.html', {'analysis_result': analysis_result})

def predict_tumor_stage(request):
    prediction = None

    # Get tumor type and size in mm² from GET parameters
    tumor_type = request.GET.get('tumor_type', '')
    tumor_size_mm2 = request.GET.get('tumor_size_cm', '')

    # Convert mm² to cm² (1 cm² = 100 mm²)
    try:
        tumor_size_cm2 = round(float(tumor_size_mm2) / 100, 2)
    except (ValueError, TypeError):
        tumor_size_cm2 = ''

    initial_data = {
        'tumor_type': tumor_type,
        'tumor_size_cm': tumor_size_cm2,
    }

    if request.method == 'POST':
        form = TumorPredictionForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            prediction = predict_stage(cd)
    else:
        form = TumorPredictionForm(initial=initial_data)

    return render(request, 'stage_predictor.html', {'form': form, 'prediction': prediction})
