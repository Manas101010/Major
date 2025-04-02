import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

saved_model = load_model("model/model.h5")
status = True

print('here', saved_model)

def check(input_img):
    print("Your image is: " + input_img)
    
    img = image.load_img("pages/static/images/" + input_img, target_size=(224, 224))
    img = image.img_to_array(img)  # Convert PIL Image to numpy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    print(img)
    
    output = saved_model.predict(img)
    
    print(output)
    
    if output[0][0] == 1:
        status = "Brain Tumor Detected"
    else:
        status = "Brain Tumor Not Detected"
    
    print(status)
    
    return status
