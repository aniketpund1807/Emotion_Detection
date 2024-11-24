import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model (update 'your_model.h5' with your model's filename)
model = tf.keras.models.load_model("model_file.keras")

# Define the mapping of class indices to labels (adjust this based on your model's classes)
class_labels = ['Angery', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']# Adjust as needed


# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((48, 48))  # Resize to model's expected input size
    image = image.convert('L')  # Convert to grayscale if needed
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale images
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Define the detection function
def detect_emotion(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_labels[predicted_class], confidence


# Streamlit Interface
st.title("Emotion Detection")

# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Detect button
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect"):
        # Perform detection
        label, confidence = detect_emotion(image)

        # Display the result
        st.write(f"**Detected Emotion:** {label}")
        #st.write(f"**Confidence:** {confidence:.2f}")
