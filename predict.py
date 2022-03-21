from enum import Enum

import dotenv
import numpy as np
import tensorflow as tf
import os
from functools import cmp_to_key
import re


class Types(Enum):
    grass = 1
    farm = 2
    factory = 3
    water = 4
    forest = 5
    building = 6
    park = 7


def dict2txt(res_dict: dict):
    def cmp(k1, k2):
        k1_name = os.path.basename(k1)
        k2_name = os.path.basename(k2)
        k1_name = int(re.search('\d+', k1_name.split(".")[0]).group(0))
        k2_name = int(re.search("\d+", k2_name.split(".")[0]).group(0))
        return k1_name - k2_name

    print(res_dict)
    res = sorted(res_dict, key=cmp_to_key(cmp))
    f2 = open(os.getenv("label_path"), 'r')
    with open(f"{os.path.basename(os.getenv('model'))}.csv", 'w') as f:
        for k in list(res):
            label = f2.readline().strip()
            s = os.path.basename(k) + "," + res_dict[k] + "," + str(Types[res_dict[k]].value) + "," + label + "\n"
            f.write(s)


def predict(model, dataset):
    print(model.name, dataset)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset,
        image_size=(int(os.getenv("img_height")), int(os.getenv("img_width"))),
        batch_size=int(os.getenv("batch_size")),
        shuffle=False)
    model.evaluate(test_ds)
    class_name = np.array(test_ds.class_names)
    a = model.predict(test_ds)
    res_dict = dict(zip(test_ds.file_paths, class_name[np.argmax(a, axis=1)]))
    dict2txt(res_dict)
    pass


def main(args):
    model = keras.models.load_model(FLAGS.model)
    dataset = FLAGS.dataset
    predict(model, dataset)


if __name__ == '__main__':
    from my_models import *

    dotenv.load_dotenv()
    FLAGS = tf.compat.v1.flags.FLAGS
    tf.compat.v1.flags.DEFINE_string("model", os.getenv("model"), "model that used")
    tf.compat.v1.flags.DEFINE_string("dataset", os.getenv("test_dataset"), "dataset for train")
    tf.compat.v1.app.run()
