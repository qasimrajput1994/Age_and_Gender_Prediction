# Age_and_Gender_Prediction
Welcome to the Age and Gender Prediction project! This repository contains a Convolutional Neural Network (CNN) model built with TensorFlow/Keras that predicts the age and gender of individuals from facial images. The model is integrated with a user-friendly Gradio interface, allowing users to upload images and receive instant predictions. Additionally, the project is deployed on Hugging Face Spaces for easy accessibility and sharing. 

**If any file is missing please visit hugging face spaces**

https://huggingface.co/spaces/mqasim686422/Prediction_Age_and_Gender/tree/main

# Demo
https://huggingface.co/spaces/mqasim686422/Prediction_Age_and_Gender
# Features
**Dual Prediction Outputs:** Predicts both age (regression) and gender (classification) from input images.

**Custom CNN Architecture:** Designed specifically for accurate feature extraction from facial images.

**Gradio Interface:** Interactive web interface for seamless user experience.

**Easy Deployment:** Hosted on Hugging Face Spaces for persistent and accessible usage.

**Comprehensive Documentation:** Detailed instructions for setup, usage, and deployment. 

# Model Details
***Architecture***

The CNN model is designed with the following layers:

**Convolutional Layers:** Multiple Conv2D layers with increasing filter sizes to capture intricate features.

**Pooling Layers:** MaxPooling2D layers to reduce spatial dimensions and control overfitting.

**Batch Normalization:** Applied after convolutional layers to stabilize and accelerate training.

**Dense Layers:** Fully connected layers for final prediction tasks.

**Dropout Layers:** Regularization layers to prevent overfitting.

# Saving and Loading

The trained model is saved in Keras format **(modelAP.keras)** and loaded within the Gradio app for making predictions.

# Dependencies
tensorflow
gradio
Pillow
numpy
matplotlib
# Data Used
**Dataset:** *UTKFace dataset from Kaggle*

*The dataset contains labeled images with age, gender, and race information. The model focuses on predicting age and gender from the provided images.*

# Acknowledgements
*TensorFlow/Keras:* For providing the deep learning framework.

*Gradio:* For the intuitive interface library.

*Hugging Face:* For hosting the application on Spaces.

# Running the Project
git clone https://github.com/qasimrajput1994/Age_and_Gender_Prediction.git

# Files in the Repository
**app.py** : Contains the code for deploying the Gradio web interface. The script loads the pre-trained model, preprocesses the input image, and provides age and gender predictions.

**modelAP.keras** : The pre-trained model saved in Keras format. This is used for inference on uploaded images.

**Age_Gender_Prediction** :Complete code file. 

**requirements.txt** : This file contains all the dependencies needed to run the project. It includes libraries such as TensorFlow, Keras, Gradio, and other required packages.
