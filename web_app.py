import streamlit as st
import os
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
#from tensorflow.preprocessing.image import load_img, img_to_array
from tensorflow.keras import models, layers
from openai import OpenAI
from dotenv import load_dotenv,find_dotenv
import re

#load the .env file

_=load_dotenv(find_dotenv())
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# defaults to getting the key using os.environ.get("OPENAI_API_KEY")
# if you saved the key under a different environment variable name, you can do something like:
# client = OpenAI(
#   api_key=os.environ.get("CUSTOM_ENV_NAME"),
# )


model = models.load_model("mymodel.h5")

labels = ['Apple', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Carambula',
 'Cherry', 'Clementine', 'Coconut', 'Cocos', 'Dates', 'Fig', 'Granadilla', 'Grape', 'Grapefruit', 'Guava',
 'Huckleberry', 'Kaki', 'Kohlrabi', 'Kumquats', 'Lychee', 'Mandarine', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo',
 'Mulberry', 'Nectarine', 'Orange', 'Papaya', 'Passion Fruit', 'Pear', 'Pepino', 'Pineapple', 'Plum', 'Pomegranate',
 'Pomelo', 'Pomelo Sweetie', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Tamarillo',
 'Tangelo', 'Watermelon', 'cocoa', 'kiwi', 'mandarin', 'mango', 'peach']

fruits = ['Apple', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Carambula',
 'Cherry', 'Clementine', 'Coconut', 'Cocos', 'Dates', 'Fig', 'Granadilla', 'Grape', 'Grapefruit', 'Guava',
 'Huckleberry', 'Kaki', 'Kohlrabi', 'Kumquats', 'Lychee', 'Mandarine', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo',
 'Mulberry', 'Nectarine', 'Orange', 'Papaya', 'Passion Fruit', 'Pear', 'Pepino', 'Pineapple', 'Plum', 'Pomegranate',
 'Pomelo', 'Pomelo Sweetie', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Tamarillo',
 'Tangelo', 'Watermelon', 'cocoa', 'kiwi', 'mandarin', 'mango', 'peach']


def fruit_details(fruit):
    completion1 = client.chat.completions.create(
        model="gpt-4-0125-preview",
        temperature=0.9,
        messages=[
            ]
            )
    fr_msg=completion1.choices[0].message.content
    return(fr_msg)

def fruit_nutrients(fruit):
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        temperature=0,
        messages=[
     ]
     )
    
    reply_text=completion.choices[0].message.content
    html_table_match = match = re.search(r'<table border="1">.*?</table>', reply_text, re.DOTALL)
    html_table = html_table_match.group(0).strip() if html_table_match else None

    return(html_table)


def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


def run():
    st.title("Fruits🍍- Classification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((224, 224))
        st.image(img, use_column_width=False)
        save_image_dir='upload_images/'
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)
        save_image_path = save_image_dir+ img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result = prepare_image(save_image_path)
            st.success("**Predicted : " + result + '**')
            with st.spinner("Looking for Fruit Facts...."):
                fruit_fact=fruit_details(result)
            if fruit_fact:
                st.info(f'**Fruit Facts of {result} :**')
                st.write(fruit_fact)
            with st.spinner("Collecting Fruit Nutrition Data...."):
                fruit_nutrient_fact=fruit_nutrients(result)
            if fruit_nutrient_fact:
                st.info(f'**Nutrition Facts of {result} : Per 100 grams Serving**')
                st.markdown(fruit_nutrient_fact, unsafe_allow_html=True)
            else : print("Sorry we Cant show the Nutrition Facts now. Try Again")



run()
#print("Everything allright")