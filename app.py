import json
import requests
import numpy as np
import streamlit as st
from urllib import response
import matplotlib.pyplot as plt

import keras
import random
import tensorflow as tf
from flask import Flask, request

URI = 'http://127.0.0.1:5000/'
model = keras.models.load_model('MNIST.h5')
feature_model = keras.models.Model(
    model.inputs, [layer.output for layer in model.layers]
)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Neural Network Visualizer')
st.sidebar.title('Input Image')
index = random.randint(0, 10000)

if st.sidebar.button('**Get Random Prediction**'):
    image = x_test[index, :, :]

    image_arr = np.reshape(image, (1, 784))
    preds = feature_model.predict(image_arr)
    preds = [p.tolist() for p in preds]

    image = image.tolist()
    image = np.reshape(image, (28, 28))
    # st.sidebar.markdown("")
    st.sidebar.image(image, width=250) 
    st.sidebar.markdown(f"Original Prediction: {y_test[index]}")
    st.sidebar.markdown("")

    for layers, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize=(50, 40))

        row = 10
        col = 10

        for i, number in enumerate(numbers):
            plt.subplot(row, col, i+1)
            plt.imshow(number*np.ones((8, 8, 3)).astype('float32'))
            plt.xticks([])
            plt.yticks([])

            if layers == len(preds)-1:
                plt.xlabel(str(i), fontsize=40)
            else:
                plt.xlabel(str(i+1), fontsize=40)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()

        if layers == len(preds)-1:
            st.text(f'Output Layer')
        else:
            st.text(f'Hidden Layer {layers+1}: Layer with {i+1} Neurons')

        st.pyplot()
    
    st.subheader(f"Model Prediction = {y_test[index]}")

st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("### *Made by: Umang Kirit Lodaya*")
st.sidebar.markdown("***[GitHub](https://github.com/Umang-Lodaya/6NN-VISUALIZER-WEB-APP) | [LinkedIn](https://www.linkedin.com/in/umang-lodaya-074496242/) | [Kaggle](https://www.kaggle.com/umanglodaya)***")
