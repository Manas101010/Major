'''import cv2
import numpy as np
import tensorflow as tf

# Load the segmentation model
segmentation_model = tf.keras.models.load_model('segmentation_model.keras')  # Segmentation model in .keras format

image_size = (256, 256)  # Image size used during training for segmentation

def preprocess_image_for_segmentation(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is not None:
        img = cv2.resize(img, image_size)  # Resize the image
        img = img / 255.0  # Normalize the image
        img = img.reshape(image_size[0], image_size[1], 1)  # Reshape to include the grayscale channel
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    else:
        raise ValueError("Error loading image. Please check the file path and format.")

def run_segmentation(img_path):
    img_array = preprocess_image_for_segmentation(img_path)
    segmentation_result = segmentation_model.predict(img_array)
    return segmentation_result  # Assuming this returns the segmented mask or analysis

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'segmentation.keras')

# Assuming your model is saved in 'segmentation_model.h5'
unet_model = tf.keras.models.load_model(model_path)

# Define image size used during training (adjust this if necessary)
image_size = (128, 128)

def preprocess_image(img_path):
    """ Preprocess the input image for the segmentation model. """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
    if img is not None:
        img = cv2.resize(img, image_size)  # Resize image to match the model input size
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to 3-channel RGB
        img = img / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    else:
        raise ValueError("Error loading image. Please check the file path.")
    


def predict_segmentation(img_path):
    """ Predict the segmentation mask for an input image. """
    img = preprocess_image(img_path)
    prediction = unet_model.predict(img)
    
    # Post-process the prediction to match the original image size
    predicted_mask = np.squeeze(prediction, axis=0)  # Remove batch dimension
    predicted_mask = np.squeeze(predicted_mask, axis=-1)  # Remove channel dimension
    
    return predicted_mask

def display_segmentation(img_path, predicted_mask):
    """ Display the original image and the predicted segmentation mask. """
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.resize(original_img, image_size)

    plt.figure(figsize=(10, 5))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    
    # Show predicted segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Segmentation Mask')
        
    plt.show()
'''
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the model path and load the segmentation model
model_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'segmentation.keras')
unet_model = tf.keras.models.load_model(model_path)

# Define the image size used during training (adjust this if necessary)
image_size = (128, 128)

def preprocess_image(img_path):
    """ Preprocess the input image for the segmentation model. """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
    if img is not None:
        img = cv2.resize(img, image_size)  # Resize image to match the model input size
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to 3-channel RGB
        img = img / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    else:
        raise ValueError("Error loading image. Please check the file path.")

def predict_segmentation(img_path):
    """ Predict the segmentation mask for an input image. """
    img = preprocess_image(img_path)
    prediction = unet_model.predict(img)
    
    # Post-process the prediction to match the original image size
    predicted_mask = np.squeeze(prediction, axis=0)  # Remove batch dimension
    predicted_mask = np.squeeze(predicted_mask, axis=-1)  # Remove channel dimension
    
    return predicted_mask

def save_segmentation(img_path, predicted_mask, output_dir='media/segmentation_results'):
    """ Save the predicted segmentation mask as an image file. """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate the output file path
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, f'segmented_{filename}').replace('\\', '/')
    print(output_path)
    
    # Save the predicted mask as an image
    plt.imsave(output_path, predicted_mask, cmap='gray')
    
    return output_path

def display_segmentation(img_path, predicted_mask):
    """ Display the original image and the predicted segmentation mask. """
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.resize(original_img, image_size)

    plt.figure(figsize=(10, 5))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    
    # Show predicted segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Segmentation Mask')
        
    plt.show()

# Example usage (commented out to avoid accidental execution when imported)
# img_path = 'path/to/your/image.jpg'
# predicted_mask = predict_segmentation(img_path)
# segmentation_path = save_segmentation(img_path, predicted_mask)
# display_segmentation(img_path, predicted_mask)

# Example usage
