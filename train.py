import logging
import os
import pickle

from typing import Tuple, Any
from collections import Counter

from mlflow.tracking import MlflowClient
import mlflow

import yaml
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score

import src


# config_path = os.path.join('/Users/maksimfomin/IT/DS_practice/4.CV/1.Similar_to_actor/config/params.yaml')
config_path = os.path.join('config/params.yaml')
config = yaml.safe_load(open('config/params.yaml'))['train']

RAND = config['random_state']
test_size = config['test_size']
SIZE = config['load']['images']['SIZE']
limit = config['load']['images']['limit_load']
key_load_img = config['key_load_img']

path_model = config['path_model']

actresses = config['load']['images']['actresses']
path_load_actresses = config['load']['path']['load_women']
path_write_actresses = config['load']['path']['write_women']

actors = config['load']['images']['actors']
path_load_actors = config['load']['path']['load_men']
path_write_actors = config['load']['path']['write_men']


# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def get_version_model(config_name, client):
    """
    Get model last version from MLFlow
    """
    dict_push = {}
    for count, value in enumerate(client.search_model_versions(f"name='{config_name}'")):
        # All model versions
        dict_push[count] = value
    return dict(list(dict_push.items())[0][1])['version']

def check_count_images(target: list) -> Tuple[int, str]:
    """
    Check count founded faces
    :param target: target
    :return: min photo count with faces, name actor/actress
    """
    check = Counter(target)
    min_item = np.inf
    name_check = ''
    for key in check.keys():
        if check[key] < min_item:
            min_item = check[key]
            name_check = key
    return min_item, name_check


def fit(random_state: int,
        test_size: int,
        embedings: np.array,
        target: list,
        path_model: str) -> tuple[Any, float, Any]:
    """
    Model fit
    :param embedings: embeddings with recognised faces
    :param test_size: test size
    :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embedings, target, test_size=test_size, stratify=target, random_state=random_state)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    with open(path_model, 'wb') as f:
        pickle.dump(model, f)
    f1_metric = f1_score(y_test, model.predict(X_test), average='macro')
    accuracy = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test), average='macro')

    print(f'F1 score = {f1_metric}')

def load_files(path_to: str) -> Tuple[np.array, list]:
    """
    Download embeddings and target to fit
    :param path_to:
    :return:
    """
    logging.info('Loading embedings & labels')
    with open(f'{path_to}/embedings.pkl', 'rb') as f:
        embedings = pickle.load(f)

    with open(f'{path_to}/labels.pkl', 'rb') as f:
        targets = pickle.load(f)
    return embedings, targets


def main():
    # If is it necessary download images from the internet
    if key_load_img:
        # Download images from the internet

        # Women
        src.load_images(path_load_actresses, actresses, limit_load=limit)
        # change image size
        src.format_images(path_load_actresses, actresses, SIZE)
        # get embeddings and save to folder
        emb = src.GetEmbedings(list_actors=actresses, path_load=path_load_actresses, path_write=path_write_actresses)
        emb.get_save_embedding()

        # Men
        src.load_images(path_load_actors, actors, limit_load=limit)
        # change image size
        src.format_images(path_load_actors, actors, SIZE)
        # get embeddings and save to folder
        emb = src.GetEmbedings(list_actors=actors, path_load=path_load_actors, path_write=path_write_actors)
        emb.get_save_embedding()

    # Open saved embeddings and dict with actresses
    embedings, target_list = load_files(path_write_actresses)
    min_item, name_check = check_count_images(target_list)

    if min_item > 1:
        logging.info('Fitting the model')

        # MLFlow tracking
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(config['name_experiment'])
        with (mlflow.start_run()):
            X_train, X_test, y_train, y_test = train_test_split(
                embedings, target_list, test_size=test_size, stratify=target_list, random_state=RAND)

            model = LogisticRegression()
            model.fit(X_train, y_train)

            with open(path_model, 'wb') as f:
                pickle.dump(model, f)

            f1_metric = f1_score(y_test, model.predict(X_test), average='macro')
            accuracy = accuracy_score(y_test, model.predict(X_test))
            precision = precision_score(y_test, model.predict(X_test), average='macro')

            print(f'F1 score = {f1_metric}')

            # Model and parameters logging
            mlflow.log_param('f1', f1_metric)
            mlflow.log_param('accuracy', accuracy)
            mlflow.log_param('precision', precision)
            mlflow.sklearn.log_model(model,
                                     artifact_path='model_lr',
                                     registered_model_name=f"{config['model_lr']}")
            mlflow.log_artifact(local_path='./train.py',
                                artifact_path='code')
            mlflow.end_run()

        # Get model last version and save to files
        client = MlflowClient()
        last_version_lr = get_version_model(config['model_lr'], client)

        yaml_file = yaml.safe_load(open(config_path))
        yaml_file['predict']["version_lr"] = int(last_version_lr)

        with open(config_path, 'w') as fp:
            yaml.dump(yaml_file, fp, encoding='UTF-8', allow_unicode=True)
    else:
        logging.info(f'Problem with size dataset {name_check}')
    return f1_metric, accuracy, precision

if __name__ == "__main__":
    main()
