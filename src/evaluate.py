import os
import glob
import argparse

import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn import metrics

parser = argparse.ArgumentParser(description="")

parser.add_argument("-m", "--model-path", type=str)
parser.add_argument("-o", "--output_path", type=str, default="./")

def evaluate_model(model_path, x_data, y_data, output_path, image_size=(28, 28), index_label={ }, label_index={ }):
    """ Evaluate keras model using standard regression metrics.
    """

    if not label_index and not index_label:
        raise Exception("")

    if index_label and not label_index:
        label_index = ""
    
    if label_index and not index_label:
        index_label = ""

    print("Evaluating %s..." % (model_path))

    model = keras.models.load_model(model_path)

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    y_pred = []
    y_true = []

    prediction_file = open("%s/%s_predictions.csv" % (output_path, model_name), "w")
    evaluation_file = open("%s/%s_evaluation.csv" % (output_path, model_name), "w")

    prediction_file.write("predicted-label")
    prediction_file.write(";actual-label")

    for class_name, _ in index_label.items():
        prediction_file.write(";%s" % class_name)
    
    prediction_file.write("\n")

    for data_index, (data_item, true_index) in enumerate(zip(x_data, y_data)):

        print("Progress - Evaluating - %s / %s" % (data_index, len(x_data)), end="\r")
        
        data_item = np.array(data_item, np.float32)

        data_item = np.expand_dims(data_item, axis=0)

        predictions = model.predict(data_item)

        if len(predictions[0]) == 1:
            top_prediction_index = int(round(predictions[0][0]))
            if top_prediction_index > len(index_label.items()) - 1:
                top_prediction_index = len(index_label.items()) - 1
            if top_prediction_index < 0:
                top_prediction_index = 0
        else:
            top_prediction_index = int(np.argmax(predictions, axis=1)[0])

        top_prediction_label = index_label[top_prediction_index]
        
        y_pred.append(top_prediction_index)
        y_true.append(index_label[true_index])

        prediction_file.write("%s" % top_prediction_label)
        prediction_file.write(";%s" % true_index)

        for prediction in predictions[0]:
            prediction_file.write(";%.2f" % prediction)

        prediction_file.write("\n")

    evaluation_file.write(np.array2string(metrics.confusion_matrix(y_true, y_pred), separator=", "))

    evaluation_file.write("--- Macro Averaged Resutls ---\n")
    evaluation_file.write("Precision: %s\n" % metrics.precision_score(y_true, y_pred, average="macro"))
    evaluation_file.write("Recall: %s\n" % metrics.recall_score(y_true, y_pred, average="macro"))
    evaluation_file.write("F1-Score: %s\n\n" % metrics.f1_score(y_true, y_pred, average="macro"))


    evaluation_file.write("--- Micro Averaged Resutls ---\n")
    evaluation_file.write("Precision: %s\n" % metrics.precision_score(y_true, y_pred, average="micro"))
    evaluation_file.write("Recall: %s\n" % metrics.recall_score(y_true, y_pred, average="micro"))
    evaluation_file.write("F1-Score: %s\n\n" % metrics.f1_score(y_true, y_pred, average="micro"))

    evaluation_file.write("--- Other Resutls ---\n")
    evaluation_file.write("MCC: %s\n" % metrics.matthews_corrcoef(y_true, y_pred))

if __name__ == "__main__":

    args = parser.parse_args()

    model_path = args.model_path
    output_path = args.output_path

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_test = x_test[..., tf.newaxis]
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    evaluate_model(
        model_path=model_path,
        x_data=x_test,
        y_data=y_test,
        output_path=output_path,
        index_label={i: i for i in range(0, 10)}
    )


