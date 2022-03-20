import os

import keras
from keras import layers
from keras.applications.resnet import ResNet101
from keras.layers import Conv2D, Flatten, Dense, Rescaling, MaxPooling2D, Dropout
from keras.models import Sequential
import tensorflow as tf
import tensorflow_hub as hub

from utils.Resnet import ResNet18


class BaseModel:
    def __init__(self):
        self.name = self.__class__.__name__
        self.model = Sequential(name=self.name)

        self.data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal",
                                  input_shape=(int(os.getenv("img_height")),
                                               int(os.getenv("img_width")),
                                               3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )

    def __call__(self, *args, **kwargs) -> Sequential:
        pass


class Vgg16(BaseModel):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        self.model.add(self.data_augmentation)
        self.model.add(Rescaling(scale=1.0 / 255))
        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(int(os.getenv("num_class")), activation='softmax'))
        self.model.summary()
        return self.model


class Resnet(BaseModel):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        model = ResNet18(int(os.getenv("num_class")))
        model.build(input_shape=(None, int(os.getenv("img_height")), int(os.getenv("img_width")), 3))
        model.summary()
        print(model.name)
        return model


class TransferModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.pre_model = os.path.join("pre_model", "")

    def __call__(self, *args, **kwargs):
        mobilenet_v2 = self.pre_model

        classifier_model = mobilenet_v2
        flag = True
        feature_extractor_layer = hub.KerasLayer(
            classifier_model,
            input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3),
            trainable=flag)

        model = tf.keras.Sequential([
            self.data_augmentation,
            feature_extractor_layer,
            keras.layers.Dense(int(os.getenv("num_class")))
        ])

        return model


class tf2_preview_mobilenet_v2_classification_4(TransferModel):
    def __init__(self):
        super(tf2_preview_mobilenet_v2_classification_4, self).__init__()
        self.pre_model = os.path.join("pre_model", 'tf2-preview_mobilenet_v2_classification_4')


class Resnet101(BaseModel):
    def __init__(self):
        super(Resnet101, self).__init__()

    def __call__(self, *args, **kwargs):
        base_model = ResNet101(weights="imagenet",
                               input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3),
                               include_top=False)
        base_model.trainable = False if os.getenv("transfer_learning_trainable") == "False" else True
        inputs = keras.Input(shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3))
        x = self.data_augmentation(inputs)
        x = Rescaling(1./255)(x)
        x = base_model(x, training=False if os.getenv("transfer_learning_trainable") == "False" else True)
        x = keras.layers.GlobalAvgPool2D()(x)
        out_puts = keras.layers.Dense(int(os.getenv("num_class")))(x)
        model = keras.Model(inputs, out_puts, name=self.name)
        return model
