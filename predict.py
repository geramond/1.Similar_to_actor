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
path_load = config['path_load']

with open('data/processed/dict_labels.json', 'r') as openfile:
    dict_labels = json.load(openfile)

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)

def process_dict_labels(dict_labels):
    list_of_empty = []
    for actress in list(dict_labels.keys()):
        if not os.listdir(f"data/raw/{actress}"):
            list_of_empty.append(actress)

    for i in list_of_empty:
        dict_labels.pop(i)
    return dict_labels


def predict_actress(image: np.array,
                    model: pickle,
                    dict_labels: dict) -> Union[None, Tuple[str, float, pd.DataFrame]]:
    """
    Predict actor/actress on photo
    :param image: test image
    :param model: model
    :param dict_labels: dict with actors/actresses names
    :return: name, proba, results frame
    """
    logging.info('Search for a face in a photo')
    face_bounding_boxes = face_recognition.face_locations(image)

    if len(face_bounding_boxes) != 1:
        print('Problem with finding a face')
    else:
        logging.info('Create bbox for a test image')
        # Transform photo with face to vector, get embedding
        face_enc = face_recognition.face_encodings(image)[0]
        # Predict actor/actress
        predict = model.predict([face_enc])
        predict_name = list(dict_labels.keys())[list(dict_labels.values()).index(predict)]
        predict_proba = model.predict_proba([face_enc])[0][predict][0]

        frame_proba = pd.DataFrame()
        frame_proba['actress'] = list(dict_labels.keys())
        frame_proba['score'] = model.predict_proba([face_enc])[0]

        return predict_name, predict_proba, frame_proba.sort_values(by='score')[::-1]
    return None


dict_labels = process_dict_labels(dict_labels)


def main():
    # Download last saved models from MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_uri_lr = f"models:/{config['model_lr']}/{config['version_lr']}"
    model = mlflow.sklearn.load_model(model_uri_lr)

    file_img = Image.open(path_load)
    image_resize = np.array(src.resize_images(file_img, size_new=SIZE))

    predict_labels, predict_value, frame_proba = predict_actress(image=image_resize,
                                                                 model=model,
                                                                 dict_labels=dict_labels)
    print(predict_labels, round(predict_value, 2), '\n')
    print(frame_proba[:5])
    print('Hello!')

if __name__ == "__main__":
    main()
