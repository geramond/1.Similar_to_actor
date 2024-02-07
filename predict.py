import os
import yaml
import json
from typing import Tuple, Union

import pickle
import logging

import mlflow

from PIL import Image
import face_recognition

import pandas as pd
import numpy as np

import src

config_path = os.path.join('config/params.yaml')
config = yaml.safe_load(open('config/params.yaml'))['predict']
path_model = config['path_model']
SIZE = config['SIZE']

path_load_actresses = config['load']['path']['load_women']
path_load_actors = config['load']['path']['load_men']

path_load = config['path_load']

# with open('data/processed/women/dict_labels.json', 'r') as openfile:
#     dict_labels = json.load(openfile)

with open('data/processed/women/dict_labels.json', 'r') as openfile:
    dict_labels_women = json.load(openfile)

with open('data/processed/men/dict_labels.json', 'r') as openfile:
    dict_labels_men = json.load(openfile)

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def process_dict_labels(dict_labels, gender):
    list_of_empty = []
    for person in list(dict_labels.keys()):
        if not os.listdir(f"data/raw/{gender}/{person}"):
            list_of_empty.append(person)

    for i in list_of_empty:
        dict_labels.pop(i)
    return dict_labels


def predict_process(image, model, dict_labels):
    logging.info('Search for a face in a photo')
    face_bounding_boxes = face_recognition.face_locations(image)

    if len(face_bounding_boxes) != 1:
        print('Problem with finding a face')
    else:
        logging.info('Create bbox for a test image')
        # Transform photo with face to vector, get embedding
        face_enc = face_recognition.face_encodings(image)[0]

        # Predict actress
        predict = model.predict([face_enc])
        predict_name = list(dict_labels.keys())[list(dict_labels.values()).index(predict)]
        predict_proba = model.predict_proba([face_enc])[0][predict][0]

        frame_proba = pd.DataFrame()
        frame_proba['actress'] = list(dict_labels.keys())
        frame_proba['score'] = model.predict_proba([face_enc])[0]

        return predict_name, predict_proba, frame_proba.sort_values(by='score')[::-1]
    return None


def predict_actress(image: np.array,
                    dict_labels: dict) -> Union[None, Tuple[str, float, pd.DataFrame]]:
    """
    Predict actor/actress on photo
    :param image: test image
    :param dict_labels: dict with actresses names
    :return: name, proba, results frame
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri_lr = f"models:/{config['model_lr_women']}/{config['version_lr_women']}"
    model = mlflow.sklearn.load_model(model_uri_lr)

    result = predict_process(image, model, dict_labels)

    return result


def predict_actor(image: np.array,
                  dict_labels: dict) -> Union[None, Tuple[str, float, pd.DataFrame]]:
    """
    Predict actor/actress on photo
    :param image: test image
    :param dict_labels: dict with actors names
    :return: name, proba, results frame
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri_lr = f"models:/{config['model_lr_men']}/{config['version_lr_men']}"
    model = mlflow.sklearn.load_model(model_uri_lr)

    result = predict_process(image, model, dict_labels)

    return result


dict_labels_women = process_dict_labels(dict_labels_women, 'women')
dict_labels_men = process_dict_labels(dict_labels_men, 'men')


def main(gender):
    file_img = Image.open(path_load)
    image_resize = np.array(src.resize_images(file_img, size_new=SIZE))

    if gender == "women":
        predict_labels, predict_value, frame_proba = predict_actress(image=image_resize,
                                                                     dict_labels=dict_labels_women)
    elif gender == "men":
        predict_labels, predict_value, frame_proba = predict_actor(image=image_resize,
                                                                   dict_labels=dict_labels_men)

    print(predict_labels, round(predict_value, 2), '\n')
    print(frame_proba[:5])
    print('Hello!')


if __name__ == "__main__":
    gender = 'women'
    main(gender)
