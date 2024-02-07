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


CONFIG_PATH = os.path.join('config/params.yaml')
CONFIG = yaml.safe_load(open('config/params.yaml'))['train']

RAND = CONFIG['random_state']
TEST_SIZE = CONFIG['test_size']
SIZE = CONFIG['load']['images']['SIZE']
LIMIT = CONFIG['load']['images']['limit_load']
KEY_LOAD_IMG = CONFIG['key_load_img']

PATH_MODEL = CONFIG['path_model']

ACTRESSES = CONFIG['load']['images']['actresses']
PATH_LOAD_ACTRESSES = CONFIG['load']['path']['load_women']
PATH_WRITE_ACTRESSES = CONFIG['load']['path']['write_women']

ACTORS = CONFIG['load']['images']['actors']
PATH_LOAD_ACTORS = CONFIG['load']['path']['load_men']
PATH_WRITE_ACTORS = CONFIG['load']['path']['write_men']


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
        embeddings: np.array,
        target: list,
        path_model: str) -> tuple[LogisticRegression, Any, float, Any]:
    """
    Model fit
    :param random_state: random state
    :param test_size: test sample size
    :param embeddings: embeddings with recognised faces
    :param target: target label
    :param path_model: model path
    :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, target, test_size=test_size, stratify=target, random_state=random_state)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    with open(path_model, 'wb') as f:
        pickle.dump(model, f)

    f1_metric = f1_score(y_test, model.predict(X_test), average='macro')
    accuracy = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test), average='macro')

    return model, f1_metric, accuracy, precision

def load_files(path_to: str) -> Tuple[np.array, list]:
    """
    Download embeddings and target to fit
    :param path_to:
    :return: embeddings, targets
    """
    logging.info('Loading embeddings & labels')
    with open(f'{path_to}/embedings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    with open(f'{path_to}/labels.pkl', 'rb') as f:
        targets = pickle.load(f)
    return embeddings, targets


def main(gender):
    # If is it necessary download images from the internet
    if KEY_LOAD_IMG:
        # Download images from the internet
        if gender == 'women':
            # Women
            src.load_images(PATH_LOAD_ACTRESSES, ACTRESSES, limit_load=LIMIT)
            # change image size
            src.format_images(PATH_LOAD_ACTRESSES, ACTRESSES, SIZE)
            # get embeddings and save to folder
            emb = src.GetEmbedings(list_actors=ACTRESSES, path_load=PATH_LOAD_ACTRESSES, path_write=PATH_WRITE_ACTRESSES)
            emb.get_save_embedding()
        elif gender == 'men':
            # Men
            src.load_images(PATH_LOAD_ACTORS, ACTORS, limit_load=LIMIT)
            # Change image size
            src.format_images(PATH_LOAD_ACTORS, ACTORS, SIZE)
            # Get embeddings and save to folder
            emb = src.GetEmbedings(list_actors=ACTORS, path_load=PATH_LOAD_ACTORS, path_write=PATH_WRITE_ACTORS)
            emb.get_save_embedding()

    # MLFlow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(CONFIG['name_experiment'] + '_' + gender)

    if gender == 'women':
        # Open saved embeddings and dict with actresses
        embeddings, target_list = load_files(PATH_WRITE_ACTRESSES)
        min_item, name_check = check_count_images(target_list)

    elif gender == 'men':
        # Open saved embeddings and dict with actors
        embeddings, target_list = load_files(PATH_WRITE_ACTORS)
        min_item, name_check = check_count_images(target_list)

    if min_item > 1:
        logging.info('Fitting the model')

        with (mlflow.start_run()):
            model, f1_metric, accuracy, precision = fit(RAND, TEST_SIZE, embeddings, target_list, PATH_MODEL)

            print(f'F1 score = {f1_metric}')

            # Model and parameters logging
            mlflow.log_param('f1', f1_metric)
            mlflow.log_param('accuracy', accuracy)
            mlflow.log_param('precision', precision)
            mlflow.sklearn.log_model(model,
                                     artifact_path='model_lr',
                                     registered_model_name=f"{CONFIG[f'model_lr_{gender}']}")
            mlflow.log_artifact(local_path='./train.py',
                                artifact_path='code')
            mlflow.end_run()

        # Get model last version and save to files
        client = MlflowClient()
        last_version_lr = get_version_model(CONFIG[f"model_lr_{gender}"], client)

        yaml_file = yaml.safe_load(open(CONFIG_PATH))
        yaml_file['predict'][f"version_lr_{gender}"] = int(last_version_lr)

        with open(CONFIG_PATH, 'w') as fp:
            yaml.dump(yaml_file, fp, encoding='UTF-8', allow_unicode=True)
    else:
        logging.info(f'Problem with size dataset {name_check}')
    return f1_metric, accuracy, precision

if __name__ == "__main__":
    for gender in ('men', 'women'):
        main(gender)
