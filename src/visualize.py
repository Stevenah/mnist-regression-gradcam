import os
import cv2
import glob
import argparse

from PIL import Image

from tensorflow import keras    
from tf_explain.core import GradCAM

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description="")

parser.add_argument("-m", "--model-paths", nargs='+')
parser.add_argument("-n", "--number-of-samples", type=int)
parser.add_argument("-o", "--output-path", type=str)

def visualize(model, image, output_path, image_name, class_index=0):
    return GradCAM().explain(
        validation_data=(image, None),
        model=model,
        layer_name="last_conv_layer",
        class_index=class_index,
        colormap=cv2.COLORMAP_JET
    )

def save_image(image, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image = Image.fromarray(image)
    image.save(path)

def extract_samples(number_of_samples, output_path):

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 28, 28)

    for class_name in range(0, 10):
        for index, image in enumerate(x_train[np.where(y_train==class_name)][:number_of_samples]):
            save_image(image, os.path.join(output_path, "train", str(class_name), "%s_%s.png" % (str(class_name), str(index))))
        for index, image in enumerate(x_test[np.where(y_test==class_name)][:number_of_samples]):
            save_image(image, os.path.join(output_path, "test", str(class_name),"%s_%s.png" % (str(class_name), str(index))))

if __name__ == "__main__":

    args = parser.parse_args()

    model_paths = args.model_paths
    number_of_samples = args.number_of_samples
    output_path = args.output_path

    sample_path = os.path.join(output_path, "samples")
    extract_samples(number_of_samples, sample_path)

    models = { }

    for model_path in model_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        models[model_name] = keras.models.load_model(model_path)

    for image_path in glob.glob(os.path.join(sample_path, "*/*/*")):

        image_dir_path = os.path.dirname(image_path)
        image_dir_path = image_dir_path.replace("samples", "visualizations")
        
        og_image = np.array(Image.open(image_path).convert("L"))

        image = np.reshape(og_image, (1, 28, 28, 1))

        image_name = image_path.split("/")[-1]
        class_name = int(image_path.split("/")[-2])

        visualizations = []

        for model_name, model in models.items():
            visualizations.append(visualize(model, image, image_dir_path, image_name))

        save_image(
            np.concatenate((np.stack((og_image,)*3, axis=-1), *visualizations)),
            os.path.join(image_dir_path, image_name)
        )
        
