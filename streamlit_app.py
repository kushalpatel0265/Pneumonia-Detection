import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

st.title("Pneumonia Detection Machine")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model.h5')
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
    
    image1 = image.load_img(image_path, target_size=(150, 150))
    image1 = image.img_to_array(image1)
    image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
    img_array = image1 / 255.0
    
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.6:
        return "You have a high chance of having Pneumonia. Please consult a doctor."
    else:
        return "You have a low chance of having Pneumonia. Nothing to panic!"

if file is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    prediction = predict(file, loaded_model)
    st.success(prediction)
