import pickle
import numpy as np
from PIL import Image

# Load the pickled model
with open('plant_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load and preprocess the image
image_path = 'sample_image.jpg'
image = Image.open(image_path)
image = image.resize((224, 224))  # Resize the image to match the model's input size
image_array = np.array(image)
image_array = image_array / 255.0  # Normalize pixel values

# Reshape the image array to match the model's expected input shape
image_input = np.expand_dims(image_array, axis=0)

# Make a prediction
predictions = model.predict(image_input)

# Assuming the model outputs class probabilities, you can find the predicted class index
predicted_class_index = np.argmax(predictions)

# Replace class_index_to_label with your mapping of class indices to disease labels
class_index_to_label = {0: 'Healthy', 1: 'Disease'}

predicted_class_label = class_index_to_label[predicted_class_index]

print(f"Predicted class: {predicted_class_label}")
print(f"Class probabilities: {predictions[0]}")
