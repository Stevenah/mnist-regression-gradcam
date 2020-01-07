
python ./src/train.py \
    --save-path ./models/regression_model.h5 \
    --output-size 1 \
    --loss mean_squared_error \
    --metrics mse \
    --activation linear;

python ./src/train.py \
    --save-path ./models/classification_model.h5 \
    --output-size 10 \
    --loss sparse_categorical_crossentropy \
    --metrics acc \
    --activation softmax;

python ./src/visualize.py \
    --model-paths ./models/regression_model.h5 ./models/classification_model.h5 \
    --number-of-samples 10 \
    --output-path ./images