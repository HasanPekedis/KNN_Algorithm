# K-Nearest Neighbors (KNN) Classifier

This project implements a K-Nearest Neighbors (KNN) classification algorithm with features such as one-hot encoding, model update, and leave-one-out cross-validation (LOOCV). The KNN algorithm can be used to classify instances based on the nearest neighbors from a training dataset.

## Features
- **Preprocessing**: Converts categorical features into numerical values using one-hot encoding.
- **Model Update**: Allows adding new training data to an existing model and saves it as a JSON file.
- **KNN Classification**: Implements KNN classification using Euclidean or Manhattan distance.
- **Leave-One-Out Cross-Validation (LOOCV)**: Evaluates model performance by leaving one instance out for testing and the rest for training.
- **Accuracy Evaluation**: Logs predicted vs. actual classifications and computes accuracy using LOOCV.

## Requirements
- Python 3.x
- `pandas`
- `numpy`
- `json`

You can install the required libraries using:

```
pip install pandas numpy
```

## Usage

1. **Load Training and Test Data**: Load datasets in JSON format (`dataset.json` and `test_dataset.json`).
2. **Preprocess Data**: Converts categorical variables to numeric using one-hot encoding.
3. **Train and Evaluate**: Choose the value of **k** (number of nearest neighbors) and select the distance calculation method (`E` for Euclidean or `M` for Manhattan).
4. **Accuracy Output**: The program will print the accuracy based on leave-one-out cross-validation and a confusion matrix.


## Example

```python
training_data = load_model("dataset.json")
test_data = load_model("test_dataset.json")
one_hot_encoded_training_data = preprocess_data(training_data)
one_hot_encoded_test_data = preprocess_data(test_data)
save_model(one_hot_encoded_training_data, "knn_model.json")

k = 3
distance_calculation_type = 'E'  # 'E' for Euclidean, 'M' for Manhattan
accuracy = leave_one_out_cross_validation(one_hot_encoded_training_data, k, distance_calculation_type)
print(f"Accuracy: {accuracy}")
```

## File Structure

- `knn_model.json`: The file that stores the processed training model.
- `classification_log.txt`: Logs predictions and actual classifications.
- `dataset.json`: The training data in JSON format.
- `test_dataset.json`: The test data in JSON format(not necessary).

## License

This project is licensed under the MIT License.
