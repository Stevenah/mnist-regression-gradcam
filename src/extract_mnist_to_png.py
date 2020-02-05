import os
import argparse

import tensorflow as tf
import numpy as np

from PIL import Image

parser = argparse.ArgumentParser(description="")

parser.add_argument("-o", "--output-path", type=str)

def save_image(image_array, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image = Image.fromarray(image_array)
    image.save(path)

def extract_images_to_png(output_path):

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 28, 28)

    for class_name in range(0, 10):
        for index, image in enumerate(x_train[np.where(y_train==class_name)]):
            save_image(image, os.path.join(output_path, "train", str(class_name), "%s_%s.png" % (str(class_name), str(index))))
        for index, image in enumerate(x_test[np.where(y_test==class_name)]):
            save_image(image, os.path.join(output_path, "test", str(class_name),"%s_%s.png" % (str(class_name), str(index))))

if __name__ == "__main__":

    args = parser.parse_args()

    output_path = args.output_path

    extract_images_to_png(output_path)

    
