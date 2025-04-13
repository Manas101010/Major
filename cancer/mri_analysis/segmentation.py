import os
import cv2
import numpy as np
import tensorflow as tf
from django.core.files.storage import default_storage
from PIL import Image
from django.conf import settings

# Register Keras custom functions
@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Load the trained model
model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'unet_brain_mri_segmentation.keras')
unet_model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient})

# Define image input size
image_size = (128, 128)

def preprocess_image(img_path):
    """ Preprocess input image for segmentation model. """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Error loading image. Please check the file path.")

    img = cv2.resize(img, image_size)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to calculate tumor size
def calculate_tumor_size(mask, pixel_area_mm2=1.0):
    tumor_pixels = np.sum(mask > 0.5)
    tumor_size_mm2 = tumor_pixels * pixel_area_mm2
    return tumor_pixels, tumor_size_mm2

def run_segmentation(img_path):
    """ Predict segmentation using U-Net model. """
    img = preprocess_image(img_path)
    # prediction = unet_model.predict(img)[0]

    # Predict segmentation mask
    pred_mask = unet_model.predict(img)[0]

    # Convert prediction to binary mask
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    # return (prediction > 0.5).astype(np.uint8) * 255
    return pred_mask,binary_mask

def visualize_segmentation(image, mask_pred):
    """
    Return two images:
    1. Original image with bounding box (BGR)
    2. Segmented tumor mask (grayscale)
    """
    mask_pred_bin = (mask_pred > 0.5).astype(np.uint8)

    # Draw bounding box on a copy
    image_with_box = image.copy()
    contours, _ = cv2.findContours(mask_pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert mask to grayscale image
    mask_gray = (mask_pred.squeeze() * 255).astype(np.uint8)

    return image_with_box, mask_gray


def process_uploaded_image(uploaded_file):
    # Save uploaded file
    file_path = default_storage.save(f'uploads/{uploaded_file.name}', uploaded_file)
    absolute_path = os.path.join(settings.MEDIA_ROOT, file_path)

    pred_mask, binary_mask = run_segmentation(absolute_path)

    original_img = cv2.imread(absolute_path)
    original_img = cv2.resize(original_img, image_size)

    image_with_box, mask_gray = visualize_segmentation(original_img, pred_mask)

    filename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    box_name = f"{filename}_box.jpg"
    mask_name = f"{filename}_mask.jpg"

    box_rel_path = os.path.join('with_bounding_box', box_name)
    mask_rel_path = os.path.join('tumor_mask', mask_name)

    box_abs_path = os.path.join(settings.MEDIA_ROOT, box_rel_path)
    mask_abs_path = os.path.join(settings.MEDIA_ROOT, mask_rel_path)

    os.makedirs(os.path.dirname(box_abs_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_abs_path), exist_ok=True)

    cv2.imwrite(box_abs_path, image_with_box)
    cv2.imwrite(mask_abs_path, mask_gray)

    tumor_pixels, tumor_size_mm2 = calculate_tumor_size(pred_mask)

    return box_rel_path, mask_rel_path, tumor_pixels, tumor_size_mm2