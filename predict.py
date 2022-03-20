import dotenv
import numpy as np
import tensorflow as tf
import os


def predict(model, dataset):
    print(model.name, dataset)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset,
        image_size=(int(os.getenv("img_height")), int(os.getenv("img_width"))),
        batch_size=int(os.getenv("batch_size")))
    model.evaluate(test_ds)
    class_name = np.array(test_ds.class_names)
    a = model.predict(test_ds)
    print(class_name[np.argmax(a, axis=1)])
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
