import os

import dotenv
import keras
import tensorflow as tf

from predict import predict
from train import train
from my_models import *


def main(args):
    if FLAGS.predict_only is False:
        model = eval(FLAGS.model)()
        model_path = train(model, FLAGS.dataset)
        model = keras.models.load_model(model_path)
        predict(model, FLAGS.test_dataset)
    else:
        model = keras.models.load_model(FLAGS.model)
        predict(model, FLAGS.test_dataset)


if __name__ == '__main__':
    dotenv.load_dotenv()
    FLAGS = tf.compat.v1.flags.FLAGS
    tf.compat.v1.flags.DEFINE_string("model", os.getenv("model"), "model that used")
    tf.compat.v1.flags.DEFINE_string("dataset", os.getenv("dataset"), "dataset for train")
    tf.compat.v1.flags.DEFINE_string('test_dataset', os.getenv("test_dataset"), "dataset for test")
    tf.compat.v1.flags.DEFINE_bool("predict_only", False if os.getenv("predict_only") == "False" else True,
                                   "train a new model or just predict a model")
    tf.compat.v1.app.run()
