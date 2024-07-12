import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.title("Pneumonia Detection Machine")

# Display current working directory for debugging
st.write(f"Current working directory: {os.getcwd()}")

# Load the model
@st.cache_resource
def load_model():
    model_path = 'saved_model/'
    if not os.path.exists(model_path):
        st.error(f"Model directory not found: {model_path}")
        return None
    try:
        model = tf.saved_model.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if 'loaded_model' not in st.session_state:
    st.session_state['loaded_model'] = load_model()

loaded_model = st.session_state['loaded_model']

# Create a temporary directory for saving uploaded files
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# File uploader
file = st.sidebar.file_uploader("Please upload your X-Ray image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

def predict(image_path, model):
    if model is None:
        return "Model not loaded."
    
    image1 = image.load_img(image_path, target_size=(150, 150))
    image1 = image.img_to_array(image1)
    image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
    img_array = image1 / 255.0
    
    # Create a serving function
    infer = model.signatures["serving_default"]
    prediction = infer(tf.constant(img_array))['dense'].numpy()
    
    if prediction[0][0] > 0.6:
        return "You have a high chance of having Pneumonia. Please consult a doctor."
    else:
        return "You have a low chance of having Pneumonia. Nothing to panic!"

if file is not None:
    try:
        img = Image.open(file).convert('RGB')
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        img_path = os.path.join(temp_dir, file.name)
        img.save(img_path)
        prediction = predict(img_path, loaded_model)
        st.success(prediction)
        os.remove(img_path)
    except Exception as e:
        st.error(f"Error processing file: {e}")
