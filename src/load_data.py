import os

import glob
import logging
import shutil

import numpy as np

from bing_image_downloader.downloader import download
from PIL import Image

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def load_images(path: str, list_actors: list, limit_load: int = 15) -> None:
    """
    Download images for model
    :param path: path to dataset
    :param list_actors: dict with names actors/actresses
    :param limit_load: download images count limit
    :return: None
    """
    logging.info('Clean the folder')
    # delete folder with dataset
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))

    logging.info('Download images of actors by Bing')
    # loop each name
    for face in list_actors:
        str_face = f'face {face}'
        # download 15 photos for current name
        download(str_face,
                 limit=limit_load,
                 output_dir=path,
                 adult_filter_off=True,
                 force_replace=False,
                 timeout=60,
                 verbose=False)
        # change folder name
        os.rename(path + '/' + str_face, path + '/' + face)
    logging.info('Completing the loading of actor images')


def resize_images(image: Image, size_new: int) -> np.array:
    """
    Image size change
    :rtype: object
    :param image: image
    :param size_new: one-side image size
    :return: image
    """
    # get image size
    size = image.size
    # get resize coeff
    coef = size_new / size[0]
    # resize image
    resized_image = image.resize(
        (int(size[0] * coef), int(size[1] * coef)))
    resized_image = resized_image.convert('RGB')
    return resized_image


def format_images(path: str, list_actors: list, size_new: int) -> None:
    """
    Reformat image size
    :param size_new: one-side image size
    :param path: path to folder with dataset
    :param list_actors: dict with names actors/actresses
    :return: None
    """
    logging.info('Formatting the image of actors')
    # loop each name
    for face in list_actors:
        # download all files name from folder
        files = glob.glob(f'{path}/{face}/*')
        # loop on files list
        for file in files:
            try:
                file_img = Image.open(file)
                resized_image = resize_images(file_img, size_new)
                resized_image.save(file)
            except Exception as ex:
                logging.info(f'Remove image {file} of {face}\nmessage: {ex}')
                os.remove(file)
