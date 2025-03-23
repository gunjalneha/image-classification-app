import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_model.keras")

model = load_model()

# Sidebar: Theme Toggle
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")


# Define theme colors
if dark_mode:
    bg_color = "#1E1E1E"  # Dark mode background
    text_color = "#FFFFFF"  # Light text
    accent_color = "#A97142"  # Brown accent
    sidebar_color = "#F9AA33"  # White sidebar in dark mode
else:
    bg_color = "#FAF3E0"  # Light beige background
    text_color = "#2B1B17"  # Dark brown text
    accent_color = "#8B5A2B"  # Darker brown accent
    sidebar_color = "#2B1B17"  # Dark brown sidebar in light mode

# Apply CSS styles
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        /* Apply font */
        * {{
            font-family: 'Poppins', sans-serif !important;
        }}
        
        /* Background color */
        .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        
        /* Titles & Text */
        h1, h2, h3, h4, h5, h6 {{
            color: {accent_color} !important;
            font-weight: 600 !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_color} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("🖼️ Image Classifier")
st.write("Upload one or multiple images, and the app will classify them.")

# File Uploader (Multiple Files Allowed)
uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        with st.spinner("🔄 Classifying... Please wait!"):
            predictions = model.predict(img_array)

        decoded_preds = decode_predictions(predictions, top=3)[0]

        # Display results
        st.subheader(f"Predictions for {uploaded_file.name}:")
        for pred in decoded_preds:
            st.write(f"**{pred[1]}** - {round(pred[2] * 100, 2)}%")

        st.markdown("---")  # Divider between images
