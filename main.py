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

CONFIG_PATH = os.path.join('/config/params.yaml')
CONFIG = yaml.safe_load(open('config/params.yaml'))['predict']
PATH_MODEL = CONFIG['path_model']
SIZE = CONFIG['SIZE']
PATH_LOAD = CONFIG['path_load']

with open('data/processed/women/dict_labels.json', 'r') as openfile:
    dict_labels_women = json.load(openfile)
dict_labels_women = predict.process_dict_labels(dict_labels_women, 'women')

with open('data/processed/men/dict_labels.json', 'r') as openfile:
    dict_labels_men = json.load(openfile)
dict_labels_men = predict.process_dict_labels(dict_labels_men, 'men')

app = FastAPI()


@st.cache_data
@app.post('/get_train')
def get_train(gender):
    f1, accuracy, precision = train.main(gender)

    result = {
        'f1': f'{f1}',
        'accuracy': f'{accuracy}',
        'precision': f'{precision}'
    }

    return result


@st.cache_data
@app.post('/get_predict')
def get_predict(image_resize, dict_labels, gender):
    if gender == "women":
        predict_labels, predict_value, frame_proba = predict.predict_actress(image=image_resize,
                                                                             dict_labels=dict_labels)
        result = {'message': 'success',
                  'predict_labels': f'{predict_labels}',
                  'predict_value': f'{predict_value}',
                  'frame_proba': f'{frame_proba}'
                  }

        return result

    elif gender == "men":
        predict_labels, predict_value, frame_proba = predict.predict_actor(image=image_resize,
                                                                           dict_labels=dict_labels)
        result = {'message': 'success',
                  'predict_labels': f'{predict_labels}',
                  'predict_value': f'{predict_value}',
                  'frame_proba': f'{frame_proba}'
                  }

        return result


def main():
    st.set_page_config(layout="wide")
    st.header('Similar actor/actress')
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        st.image(img_file_buffer)
        file_img = Image.open(img_file_buffer)
    else:
        file_img = Image.open(PATH_LOAD)
    image_resize = np.array(src.resize_images(file_img, size_new=SIZE))

    st.markdown(
        """
        ### Make photo and get Similar-to-actor predict
        ### Or get Similar-to-actor predict from test_image
        """
    )

    button_train_women = st.button("Train women")
    if button_train_women:
        result_train = get_train('women')
        st.success(f'{result_train}')

    button_train_men = st.button("Train men")
    if button_train_men:
        result_train = get_train('men')
        st.success(f'{result_train}')

    button_predict_women = st.button("Predict women")
    if button_predict_women:
        result_predict = get_predict(image_resize, dict_labels_women, 'women')

        dir = f"data/raw/women/{result_predict['predict_labels']}/"
        path_sample = dir + random.choice(os.listdir(dir))

        image_actress_sample = Image.open(path_sample)
        st.success(f'{result_predict}')
        if not img_file_buffer:
            st.image(image_actress_sample)
        else:
            st.image([img_file_buffer, image_actress_sample], width=750)

    button_predict_men = st.button("Predict men")
    if button_predict_men:
        result_predict = get_predict(image_resize, dict_labels_men, 'men')

        dir = f"data/raw/men/{result_predict['predict_labels']}/"
        path_sample = dir + random.choice(os.listdir(dir))

        image_actor_sample = Image.open(path_sample)
        st.success(f'{result_predict}')
        if not img_file_buffer:
            st.image(image_actor_sample)
        else:
            st.image([img_file_buffer, image_actor_sample], width=750)

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
#   - SOLVE ISSUE WITH PREDICT_SCORE, FRAME_PROBA
#   - REFORMAT CODE ACCORDING PEP8
