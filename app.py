import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model("blood_type_model.h5")
class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+']

# Title and description
st.title("🩸 Blood Type Detection")
st.write("Upload a scanned blood sample image to detect the blood group.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.markdown(f"### 🧬 Predicted Blood Group: `{predicted_class}`")
    st.markdown(f"Confidence: `{confidence:.2f}%`")
