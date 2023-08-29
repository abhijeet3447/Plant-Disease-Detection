import h5py
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from  tensorflow import keras
import pickle
import streamlit as st

model1 = keras.models.load_model('model_1.h5')

model2 = keras.models.load_model('model_1.h5')


diseases = ['Potato___Hollow_Heart',
 'Squash___Powdery_mildew',
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Tomato___Late_blight',
 'Strawberry___Leaf_scorch',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Early_blight',
 'Tomato___Tomato_mosaic_virus',
 'Potato___Late_blight',
 'Tomato___healthy',
 'Grape___healthy',
 'Grape___Black_rot',
 'Pepper,_bell___healthy',
 'Tomato___Canker',
 'Corn_(maize)___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Peach___healthy',
 'Soybean___healthy',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Apple___Rotten',
 'Corn_(maize)___Common_rust_',
 'Tomato___Septoria_leaf_spot',
 'Grape___Esca_(Black_Measles)',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Tea__Black_rot',
 'Potato___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Peach___Bacterial_spot',
 'Raspberry___healthy',
 'Blueberry___healthy',
 'Tea__Healthy',
 'Tomato___Leaf_Mold',
 'Tomato___Bacterial_spot',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Ginger__Healthy',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Target_Spot',
 'Strawberry___healthy',
 'Potato___Early_blight']

image_path = "TeaHealthy1.JPG"
new_img =keras.utils.load_img(image_path, target_size=(256, 256))
img = keras.utils.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
img = img/255
prediction = model1.predict(img)
#probabilty = prediction.flatten()
#max_prob = probabilty.max()
index=prediction.argmax(axis=-1)[0]
class_name = diseases[index]
#ploting image with predicted class name        
#plt.figure(figsize = (4,4))
#plt.imshow(new_img)
#plt.axis('off')
#plt.title(class_name+" "+ str(max_prob)[0:4]+"%")
#plt.show()
img_name = image_path.split('/')[-1][:-5]
print("Actual class name :", img_name)
print("Predicted class name :", class_name)




