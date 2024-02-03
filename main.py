import os
import yaml
import json
import random

import numpy as np

import pickle
from PIL import Image

import mlflow
from fastapi import FastAPI
import streamlit as st
from pydantic import BaseModel
import uvicorn

import src
import train
import predict


config_path = os.path.join('/config/params.yaml')
config = yaml.safe_load(open('config/params.yaml'))['predict']
path_model = config['path_model']
SIZE = config['SIZE']
path_load = config['path_load']

with open('data/processed/dict_labels.json', 'r') as openfile:
    dict_labels = json.load(openfile)

dict_labels = predict.process_dict_labels(dict_labels)

app = FastAPI()


@st.cache_data
@app.post('/get_train')
def get_train():
    f1, accuracy, precision = train.main()

    result = {
        'f1': f'{f1}',
        'accuracy': f'{accuracy}',
        'precision': f'{precision}'
    }

    return result


@st.cache_data
@app.post('/get_predict')
def get_predict(image_resize):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri_lr = f"models:/{config['model_lr']}/{config['version_lr']}"
    model = mlflow.sklearn.load_model(model_uri_lr)
    predict_labels, predict_value, frame_proba = predict.predict_actress(image=image_resize,
                                                                         model=model,
                                                                         dict_labels=dict_labels)
    result ={'message': 'success',
            'predict_labels':f'{predict_labels}',
            'predict_value':f'{predict_value}',
            'frame_proba':f'{frame_proba}'
            }
    return result


def main():
    st.header('Similar actor/actress')
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        st.image(img_file_buffer)
        file_img = Image.open(img_file_buffer)
    else:
        file_img = Image.open(path_load)
    image_resize = np.array(src.resize_images(file_img, size_new=SIZE))

    st.markdown(
        """
        ### Make photo and get Similar-to-actor predict
        ### Or get Similar-to-actor predict from test_image
        """
    )

    button_train = st.button("Train")
    if button_train:
        result_train = get_train()
        st.success(f'{result_train}')

    button_predict = st.button("Predict")
    if button_predict:
        result_predict = get_predict(image_resize)

        dir = f"data/raw/{result_predict['predict_labels']}/"
        path_sample = dir + random.choice(os.listdir(dir))

        image_actress_sample = Image.open(path_sample)
        st.success(f'{result_predict}')
        if not img_file_buffer:
            st.image(image_actress_sample)
        else:
            st.image([img_file_buffer, image_actress_sample], width=750)

    button_mlflow = st.button("MLFlow")
    if button_mlflow:
        mlflow_cmd = "mlflow server --host localhost --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow"
        os.system(f"{mlflow_cmd}")

    # button_fastapi = st.button("FastAPI")
    # if button_fastapi:
    #     fastapi_cmd = "python3 -m uvicorn main:app --host=127.0.0.1 --port 8000 --reload"
    #     os.system(f"{fastapi_cmd}")

    # button_airflow = st.button("Airflow")
    # if button_airflow:
    #     airflow_cmd = "export AIRFLOW_HOME=~/IT/DS_practice/4.CV/1.Similar_to_actor/airflow"
    #     os.system(f"{airflow_cmd}")
    #     airflow_cmd = "airflow webserver -p 8080"
    #     os.system(f"{airflow_cmd}")
    #     airflow_cmd = "airflow scheduler"
    #     os.system(f"{airflow_cmd}")

    # button_docker = st.button("Docker")
    # if button_docker:
    #     docker_cmd = "docker build -t similar_to_actor:latest ."
    #     os.system(f"{docker_cmd}")


if __name__ == '__main__':
    main()

# TODO
# - MEN/WOMEN RECOGNITION
# - 2 IMAGES IN 1 ROW (width=50%)
# - REFORMAT CODE ACCORDING PEP8
