from tensorflow import keras
import tensorflow as tf



def build_regression_model(input_shape):
    return keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation="linear")
    ])

def build_classification_model(input_shape):
    return keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax")
    ])

def train_model(model, x_train, y_train, loss, metrics, model_save_path):

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
            keras.callbacks.ModelCheckpoint(model_save_path)
        ]
    )

if __name__ == "__main__":
    
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    train_model(
        build_classification_model((28, 28, 1)),
        x_train,
        y_train,
        "sparse_categorical_crossentropy",
        ["acc"],
        "classification_model.h5"
    )

    tf.reset_default_graph()

    train_model(
        build_regression_model((28, 28, 1)),
        x_train,
        y_train,
        "mean_squared_error",
        ["mse"],
        "regression_model.h5"
    )




