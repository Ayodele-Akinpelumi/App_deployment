import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from tensorflow.keras import preprocessing

# Model 
def model_predictions(test_img):
    model = models.load_model(r"C:\Users\USER\Documents\Projct\trained_model.h5")
    
    # Load image using PIL
    img = Image.open(test_img)
    img = img.resize((150, 150))  # Resize image to the input size of the model
    input_array = np.array(img)  # Convert to numpy array
    input_array = np.expand_dims(input_array, axis=0)  # Convert single image into batch
    input_array = input_array / 255.0  # Normalize image

    # Make a prediction
    prediction = model.predict(input_array)
    return np.argmax(prediction)  # Return the index of the maximum value

# Load class labels from file
def load_labels(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]

# Streamlit sidebar
st.sidebar.title("Dashboard")
menu = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Predictions"])

# Main Page
if menu == "Home":
    st.header("FRUITS AND VEGETABLES RECOGNITION SYSTEM")
    # Load the image
    image = Image.open(r"C:\Users\USER\Downloads\dataset-cover.jpg")
    # Display the image
    st.image(image, caption='Welcome to My Streamlit App', use_column_width=True)

# About project
elif menu == "About Project":
    st.header("About the Project")
    st.subheader("This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks.")
    st.code("Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango")
    st.code("Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")
    st.code("This project is aimed at creating an advanced fruit and vegetable recognition system using Convolutional Neural Networks (CNNs) and TensorFlow. The system will be trained on a large dataset of images of fruits and vegetables, and it will be able to accurately identify the type of each image.")
    st.subheader("Content")
    st.text("The dataset is organized into three main folders:")
    st.text("Train: Contains 100 images per category.")
    st.text("Test: Contains 10 images per category")
    st.text("Validation: Contains 10 images per category.")

# Predictions
elif menu == "Predictions":
    st.header("Predictions")
    st.subheader("Please upload an image of a fruit or vegetable.")
    test_img = st.file_uploader("Choose an image", type=["jpg", "png"])

    if test_img is not None:
        image = Image.open(test_img)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Prediction button
        if st.button("Predict"):
            results_index = model_predictions(test_img)
            
            # Reading label
            try:
                labels = load_labels(r"C:\Users\USER\Documents\Projct\label.txt")
                st.success(f"Model is predicting, it's {labels[results_index]}")
            except Exception as e:
                st.error(f"Error loading labels: {e}")
