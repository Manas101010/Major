'''from django.shortcuts import render, redirect
from .models import MRIImage, AnalysisResult
from .classification import make_prediction  # Import classification logic
from .segmentation import run_segmentation   # Import segmentation logic

def analyze_mri(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['mri_image']
        mri_image = MRIImage.objects.create(image=uploaded_file)
        
        # Tumor detection using the first model (classification)
        prediction_label, _ = make_prediction(mri_image.image.path)
        
        tumor_detected = prediction_label != 'notumor'  # 'notumor' means no tumor detected
        
        analysis_result = AnalysisResult.objects.create(
            mri_image=mri_image,
            tumor_detected=tumor_detected
        )
        
        if tumor_detected:
            # Further analysis using the second model (segmentation)
            detailed_result = run_segmentation(mri_image.image.path)
            analysis_result.detailed_analysis = f"Segmentation result: {detailed_result}"
            analysis_result.save()
        
        return redirect('result_view', pk=analysis_result.pk)
    
    return render(request, 'upload_mri.html')

def result_view(request, pk):
    analysis_result = AnalysisResult.objects.get(pk=pk)
    return render(request, 'result_view.html', {'analysis_result': analysis_result})

from django.shortcuts import render, redirect
from .models import MRIImage, AnalysisResult
from .classification import make_prediction  # Import classification logic
from .segmentation import predict_segmentation   


def analyze_mri(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['mri_image']
        mri_image = MRIImage.objects.create(image=uploaded_file)
        
        # Tumor detection using the first model (classification)
        print('Messi')
        prediction_label, _ = make_prediction(mri_image.image.path)
        
        # Print statement to show the prediction result
        print(f"Prediction Label: {prediction_label}")

        
        tumor_detected = prediction_label != 'notumor'  # 'notumor' means no tumor detected
        
        # Print whether a tumor was detected or not
        print(f"Tumor Detected: {tumor_detected}")
        
        # Save the result to the database
        analysis_result = AnalysisResult.objects.create(
            mri_image=mri_image,
            tumor_detected=tumor_detected,
            detailed_analysis=prediction_label
        )
        
        # Comment out the segmentation part for now
        if tumor_detected:

         # Further analysis using the second model (segmentation)
            detailed_result = predict_segmentation(mri_image.image.path)
            analysis_result = f"Segmentation result: {detailed_result}"
            print(analysis_result)
            #analysis_result.save()
        
        return redirect('result_view', pk=analysis_result.pk)
    
    return render(request, 'upload_mri.html')

from django.shortcuts import render, redirect
from .models import MRIImage, AnalysisResult
from .classification import make_prediction  # Import classification logic
from .segmentation import predict_segmentation, display_segmentation  # Import segmentation functions

def analyze_mri(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['mri_image']
        mri_image = MRIImage.objects.create(image=uploaded_file)
        
        # Tumor detection using the first model (classification)
        prediction_label, _ = make_prediction(mri_image.image.path)
        
        # Print statement to show the prediction result
        print(f"Prediction Label: {prediction_label}")

        tumor_detected = prediction_label != 'notumor'  # 'notumor' means no tumor detected
        
        # Print whether a tumor was detected or not
        print(f"Tumor Detected: {tumor_detected}")
        
        # Save the result to the database
        analysis_result = AnalysisResult.objects.create(
            mri_image=mri_image,
            tumor_detected=tumor_detected,
            detailed_analysis=prediction_label
        )
        
        # Perform segmentation if a tumor is detected
        if tumor_detected:
            predicted_mask = predict_segmentation(mri_image.image.path)
            print(display_segmentation(mri_image.image.path, predicted_mask))
            analysis_result.detailed_analysis = "Segmentation performed successfully."
            analysis_result.save()
        
        return redirect('result_view', pk=analysis_result.pk)
    
    return render(request, 'upload_mri.html')

def result_view(request, pk):
    analysis_result = AnalysisResult.objects.get(pk=pk)
    
    # Print statement to show the result details
    print(f"Analysis Result ID: {analysis_result.pk}")
    print(f"Tumor Detected (in result view): {analysis_result.tumor_detected}")
    
    return render(request, 'result_view.html', {'analysis_result': analysis_result})
'''
from django.shortcuts import render, redirect
from .models import MRIImage, AnalysisResult
from .classification import make_prediction  # Import classification logic
from .segmentation import predict_segmentation, save_segmentation  # Import segmentation functions

def analyze_mri(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['mri_image']
        mri_image = MRIImage.objects.create(image=uploaded_file)
        
        
        # Tumor detection using the first model (classification)
        prediction_label, _ = make_prediction(mri_image.image.path)
        
        # Print statement to show the prediction result
        print(f"Prediction Label: {prediction_label}")

        tumor_detected = prediction_label != 'notumor'  # 'notumor' means no tumor detected
        
        # Print whether a tumor was detected or not
        print(f"Tumor Detected: {tumor_detected}")
        
        # Save the result to the database
        analysis_result = AnalysisResult.objects.create(
            mri_image=mri_image,
            tumor_detected=tumor_detected,
            detailed_analysis=prediction_label
        )
        
        # Perform segmentation if a tumor is detected
        if tumor_detected:
            predicted_mask = predict_segmentation(mri_image.image.path)
            
            # Save the segmentation mask as an image and get its path
            segmentation_path = save_segmentation(mri_image.image.path, predicted_mask)
            print('******************************************************************')
            print(segmentation_path)
            print('Manas')
            # Update the analysis result with the segmentation details

            analysis_result.detailed_analysis = prediction_label
            analysis_result.segmented_result = segmentation_path.replace('media/','')  # Assuming you have a field for the segmentation image in your model
            print(analysis_result.segmented_result)
            analysis_result.save()
        
        return redirect('result_view', pk=analysis_result.pk)
    
    return render(request, 'upload_mri.html')

def result_view(request, pk):
    analysis_result = AnalysisResult.objects.get(pk=pk)
    
    # Print statement to show the result details
    print(f"Analysis Result ID: {analysis_result.pk}")
    print(f"Tumor Detected (in result view): {analysis_result.tumor_detected}")
    
    return render(request, 'result_view.html', {'analysis_result': analysis_result})
