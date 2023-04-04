import random
import numpy as np

import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import streamlit as st
# st.set_page_config(layout="wide")

print("************************")
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
    st.sidebar.markdown(f"Original Label: {y_test[index]}")
    st.sidebar.markdown("")
    
    with st.spinner('Predicting'):
        col1, col2, col3, col4, col5, col6 = st.columns(6, gap="large")

        # st.markdown(
        #     """
        #     <style>
        #         div[data-testid="column"]
        #         {
        #             border:1px solid red;
        #             display: flex;
        #             flex-direction: column;
        #             justify-content: center;

        #         }  
        #     </style>
        #     """, unsafe_allow_html=True
        # )
        row, col = 25, 1
        
        with col1:
            layers = 0
            p = preds[0]
            numbers = np.squeeze(np.array(p))
            plt.figure(figsize=(2, 10))

            for i, number in enumerate(numbers):
                # st.write(i)
                plt.subplot(row, col, i+1)
                plt.imshow(number*np.ones((8, 8, 3)).astype('float32'))
                plt.xticks([])
                plt.yticks([])
                plt.ylabel("*", fontsize=5)

            st.pyplot()

        with col2:
            layers = 1
            p = preds[1]
            numbers = np.squeeze(np.array(p))
            plt.figure(figsize=(2, 10))

            for i, number in enumerate(numbers):
                # st.write(i)
                plt.subplot(row, col, i+1)
                plt.imshow(number*np.ones((8, 8, 3)).astype('float32'))
                plt.xticks([])
                plt.yticks([])
                plt.ylabel("*", fontsize=5)

            st.pyplot()

        with col3:
            layers = 2
            p = preds[2]
            numbers = np.squeeze(np.array(p))
            plt.figure(figsize=(2, 10))

            for i, number in enumerate(numbers):
                # st.write(i)
                plt.subplot(row, col, i+1)
                plt.imshow(number*np.ones((8, 8, 3)).astype('float32'))
                plt.xticks([])
                plt.yticks([])
                plt.ylabel("*", fontsize=5)

            st.pyplot()

        with col4:
            layers = 3
            p = preds[3]
            numbers = np.squeeze(np.array(p))
            plt.figure(figsize=(2, 10))

            for i, number in enumerate(numbers):
                # st.write(i)
                plt.subplot(row, col, i+1)
                plt.imshow(number*np.ones((8, 8, 3)).astype('float32'))
                plt.xticks([])
                plt.yticks([])
                plt.ylabel("*", fontsize=5)

            st.pyplot()

        with col5:
            layers = 4
            p = preds[4]
            numbers = np.squeeze(np.array(p))
            plt.figure(figsize=(2, 10))

            for i, number in enumerate(numbers):
                # st.write(i)
                plt.subplot(row, col, i+1)
                plt.imshow(number*np.ones((8, 8, 3)).astype('float32'))
                plt.xticks([])
                plt.yticks([])
                plt.ylabel("*", fontsize=5)

            st.pyplot()

        with col6:
            layers = 5
            p = preds[5]
            numbers = np.squeeze(np.array(p))
            plt.figure(figsize=(2, 10))

            for i, number in enumerate(numbers):
                # st.write(i)
                plt.subplot(row, col, i+1)
                plt.imshow(number*np.ones((8, 8, 3)).astype('float32'))
                plt.xticks([])
                plt.yticks([])
                plt.ylabel(str(i), fontsize=5)

            st.pyplot()
        
        st.subheader(f"Model Prediction = {y_test[index]}")

st.sidebar.markdown("")
# st.sidebar.markdown("")
# st.sidebar.markdown("")
st.sidebar.markdown("### *Made by: Umang Kirit Lodaya*")
st.sidebar.markdown("***[GitHub](https://github.com/Umang-Lodaya/6NN-VISUALIZER-WEB-APP) | [LinkedIn](https://www.linkedin.com/in/umang-lodaya-074496242/) | [Kaggle](https://www.kaggle.com/umanglodaya)***")
