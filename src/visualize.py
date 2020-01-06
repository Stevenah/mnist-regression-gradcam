
import cv2

from tensorflow import keras    
from tf_explain.core import GradCAM

def visualize(model_path, image, output_path, class_index=0):
    
    model = keras.models.load_model(model_path)

    explainer = GradCAM()

    output = explainer.explain(
        validation_data=(image, None),
        model=model,
        layer_name="",
        class_index=0,
        colormap=cv2.COLORMAP_JET
    )

    explainer.save(output, output_path, "")

    return output