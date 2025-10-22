from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("brain_tumor_model.h5")

# Path to new image
img_path = r"C:\Users\LENOVO\OneDrive\Desktop\brain\test\test3.png"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("Prediction: Yes, Tumor detected")
else:
    print("Prediction: No tumor")

# python -m venv venv
# .\venv\Scripts\activate
# python predict.py