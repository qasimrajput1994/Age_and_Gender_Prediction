import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the saved model
model = load_model('./modelAP.keras')

# Mapping gender prediction to labels
gender_mapping = {0: 'Male', 1: 'Female'}

# Target size for the model input
target_size = (128, 128)

def predict_age_gender(image):
    # Open the image and convert to grayscale
    img = Image.open(image).convert('L')
    img = img.resize(target_size)
    
    # Convert image to array and preprocess
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict age and gender
    pred = model.predict(img_array)
    predicted_gender = gender_mapping[round(pred[0][0][0])]
    predicted_age = round(pred[1][0][0])
    
    return predicted_age, predicted_gender

# Create Gradio interface
iface = gr.Interface(
    fn=predict_age_gender,
    inputs=gr.Image(type='filepath', label='Upload an Image'),
    outputs=[gr.Textbox(label='Predicted Age'), gr.Textbox(label='Predicted Gender')],
    title='Age and Gender Prediction',
    description='Upload an image to get predictions for age and gender.'
)

# Launch the interface
iface.launch()
