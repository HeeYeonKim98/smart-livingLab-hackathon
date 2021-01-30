import os
import sys
import numpy as np
import random
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import get_config
import tensorflow.keras as keras
from skimage.transform import resize
config = get_config()

def prepare_data(data_path, ctrl):
    with open(ctrl, 'r') as f:
        files = f.read().splitlines()
    data_shuffle, labels_shuffle = [], []
    data_range = list(range(len(files)))
    random.shuffle(data_range)

    for s in data_range:
        filename = files[s] + '.jpg'
        load_path = os.path.join(data_path, filename)
        image = tf.keras.preprocessing.image.load_img(load_path)
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = resize(input_arr, (64, 64, 3))
        input_arr = np.array(input_arr, dtype=np.int32)

        # fix_frame process - Can be modified

        data_shuffle.append(input_arr)
        label = files[s].split('_')[2]
        labels_shuffle.append(int(label))

    labels_shuffle = np.eye(config.n_class)[labels_shuffle]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)