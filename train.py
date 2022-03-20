import datetime
import os

import dotenv
import tensorflow as tf
from keras.optimizer_v2.adam import Adam


def train(model, dataset) -> str:
    print(f"train {model.name} {dataset}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(int(os.getenv("img_height")), int(os.getenv("img_width"))),
        batch_size=int(os.getenv("batch_size")))

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(int(os.getenv("img_height")), int(os.getenv("img_width"))),
        batch_size=int(os.getenv("batch_size")))

    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    model = model(train_ds)
    print(model.name)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(os.getenv("initial_learning_rate")),
        decay_steps=int(os.getenv("decay_steps")),
        decay_rate=float(os.getenv("decay_rate"))
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=Adam(0.002),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  # loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = os.path.join(os.getenv("model_dir"), model.name, f"{os.getenv('epochs')}-{os.getenv('batch_size')}-{os.getenv('initial_learning_rate')}-{os.getenv('decay_steps')}-{os.getenv('decay_rate')}-cp_best.ckpt")
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=False,
        save_best_only=True
    )
    log_dir = os.path.join(os.getenv("log_dir"), model.name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)  # Enable histogram computation for every epoch.

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(os.getenv("epochs")),
        callbacks=[save_callback, tensorboard_callback]
    )
    export_path = os.path.join(os.getenv("model_dir"), model.name, f"{os.getenv('epochs')}-{os.getenv('batch_size')}-{os.getenv('initial_learning_rate')}-{os.getenv('decay_steps')}-{os.getenv('decay_rate')}-best")
    model.save(export_path)
    print(export_path)
    return export_path


def main(args):
    model = eval(FLAGS.model)()
    dataset = FLAGS.dataset
    train(model, dataset)


if __name__ == '__main__':
    from my_models import *

    dotenv.load_dotenv()
    FLAGS = tf.compat.v1.flags.FLAGS
    tf.compat.v1.flags.DEFINE_string("model", os.getenv("model"), "model that used")
    tf.compat.v1.flags.DEFINE_string("dataset", os.getenv("dataset"), "dataset for train")
    tf.compat.v1.app.run()
