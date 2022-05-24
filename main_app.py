#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('dog_breed.h5') #Loading the Model
CLASS_NAMES = ['Scottish Deerhound','Maltese Dog','Bernese Mountain Dog'] #Name of Classes

st.title("Dog Breed Prediction") #Setting Title of App
st.markdown("Upload an image of the dog")


dog_image = st.file_uploader("Choose an image...", type="png") #Uploading the dog image
submit = st.button('Predict')

if submit: #  On predict button click

    if dog_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")   # Displaying the image
        opencv_image = cv2.resize(opencv_image, (224,224)) # Resizing the image
        opencv_image.shape = (1,224,224,3) # Convert image to 4 Dimension
        Y_pred = model.predict(opencv_image) # Make Prediction

        st.title(str("The Dog Breed is "+CLASS_NAMES[np.argmax(Y_pred)]))
