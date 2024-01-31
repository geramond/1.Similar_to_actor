import os
import yaml
import json

import pickle
from PIL import Image

import src
import train
import predict

import mlflow

import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


config_path = os.path.join('/Users/maksimfomin/IT/DS_practice/4.CV/Similar_to_actor/config/params.yaml')
config = yaml.safe_load(open('config/params.yaml'))['predict']
path_model = config['path_model']
SIZE = config['SIZE']
path_load = config['path_load']

with open('data/processed/dict_labels.json', 'r') as openfile:
    dict_labels = json.load(openfile)

dict_labels = predict.process_dict_labels(dict_labels)

app = FastAPI()


class ImageToPredict(BaseModel):
    path_load: str
    # test_image:

@app.post('/get_train')
def get_train():
    f1, accuracy, precision = train.main()

    result = {
        'f1': f'{f1}',
        'accuracy': f'{accuracy}',
        'precision': f'{precision}'
    }

    return result


@app.post('/get_predict')
def get_predict(test_image: ImageToPredict):
    file_img = Image.open(test_image.path_load)
    image_resize = np.array(src.resize_images(file_img, size_new=SIZE))

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

if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)