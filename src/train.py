import os
import argparse

import tensorflow as tf

from tensorflow import keras


parser = argparse.ArgumentParser(description="")

parser.add_argument("-s", "--save-path", type=str)
parser.add_argument("-o", "--output-size", type=int)
parser.add_argument("-l", "--loss", type=str)
parser.add_argument("-m", "--metrics", nargs="+")
parser.add_argument("-a", "--activation", type=str)

def build_model(input_shape, output_size, last_activation):
    return keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        keras.layers.Conv2D(64, (3, 3), activation="relu", name="last_conv_layer"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_size, activation=last_activation)
    ])

def train_model(model, x_train, y_train, loss, metrics, model_save_path):

    model_name = os.path.basename(model_save_path)

    model.compile(
        optimizer="adam",
        metrics=metrics,
        loss=loss,
    )

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            keras.callbacks.ModelCheckpoint(model_save_path)
        ]
    )

    model.save(os.path.join(os.path.dirname(model_save_path), "final_%s" % model_name))

if __name__ == "__main__":

    args = parser.parse_args()

    save_path = args.save_path
    output_size = args.output_size
    loss = args.loss
    metrics = args.metrics
    activation = args.activation

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    train_model(
        build_model((28, 28, 1), output_size, activation),
        x_train,
        y_train,
        loss,
        metrics,
        save_path   
    )


