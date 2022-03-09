import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Dermatology Image Classification Group 30')

st.markdown("Welcome to web application that classifies skin lesions that have been separated into seven categories :  Actinic keratoses, Basal cell carcinoma, Benign keratosis-like lesions , Dermatofibroma, Melanocytic nevi, Vascular lesions, Melanoma")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "derma_model.h5"
    IMAGE_SHAPE = (28, 28,3)
    model = load_model(classifier_model,compile=False,)
    test_image = image.resize((32,32))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['akiec',
          'bcc',
          'bkl',
          'df',
          'nv',
          'vasc',
          'mel']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'akiec': 0,
    'bcc': 0,
    'bkl': 0,
    'df': 0,
    'nv': 0,
    'vasc': 0,
    'mel': 0
    }


    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result









    

if __name__ == "__main__":
    main()

