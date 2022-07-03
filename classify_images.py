# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 10:13:43 2022


!pip3 install tensorflow
!pip3 install requests
!pip install gradio

@author: manish
"""
import tensorflow as tf
import requests
import gradio as gr

inception_net = tf.keras.applications.MobileNetV2()

# Download human-readable labels for ImageNet ( 1001 labels)
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def classify_images(inp):
  inp = inp.reshape((-1, 224, 224, 3))
  inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
  prediction = inception_net.predict(inp).flatten()
  confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
  return confidences

app = gr.Interface(fn=classify_images, 
                   inputs=gr.inputs.Image(shape=(224, 224)), 
                   outputs=gr.outputs.Label(num_top_classes=3))

app.launch()