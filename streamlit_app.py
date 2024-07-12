import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Increase Streamlit's file upload limit
st.set_option('server.maxUploadSize', 1024)

st.title("Pneumonia Detection Machine")

# Display current working directory for debugging
st.write(f"Current working directory: {os.getcwd()}")

# Load the model
@st.cache_resource
def load_model():
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

loaded_model = load_model()

# File uploader
file = st.sidebar.file_uploader("Please upload your X-Ray image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

def predict(image_path, model):
    if model is None:
        return "Model not loaded."
    
    try:
        image1 = image.load_img(image_path, target_size=(150, 150))
        image1 = image.img_to_array(image1)
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
        img_array = image1 / 255.0
        
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.6:
            return "You have a high chance of having Pneumonia. Please consult a doctor."
        else:
            return "You have a low chance of having Pneumonia. Nothing to panic!"
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction error."

if file is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    prediction = predict(file, loaded_model)
    st.success(prediction)
