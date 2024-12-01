```markdown
# K-Nearest Neighbors (k-NN) Classifier

## Overview

This project implements a K-Nearest Neighbors (k-NN) classifier to predict whether tennis will be played based on weather conditions. The dataset includes features such as Outlook, Temperature, Humidity, and Wind, with the target variable being `PlayTennis`. The classifier is implemented with both Euclidean and Manhattan distance metrics, and it evaluates the performance using a leave-one-out cross-validation approach.

## Requirements

To run this project, you need the following Python libraries:

- `pandas`
- `numpy`
- `json`

You can install these libraries using pip:

```bash
pip install pandas numpy
```

## Files

- `knn_classifier.py`: The main Python script that contains the k-NN algorithm implementation, data processing, and evaluation functions.
- `dataset.json`: The dataset used for training and testing the k-NN model (in JSON format).
- `test_dataset.json`: The dataset used for testing the model (in JSON format).

## Key Functions

### 1. `preprocess_data(dataset)`
Converts categorical data to numeric using one-hot encoding. This is applied to the 'Outlook', 'Temperature', 'Humidity', and 'Wind' columns.

### 2. `save_model(data, filename="knn_model.json")`
Saves the preprocessed dataset to a JSON file.

### 3. `update_model(new_data, filename="knn_model.json")`
Updates the model with new data and saves it to a JSON file.

### 4. `load_model(filename)`
Loads the preprocessed dataset from a JSON file.

### 5. `calculate_distance(instance1, instance2)`
Calculates the distance between two data points. You can use either the Euclidean distance or the Manhattan distance (based on implementation).

### 6. `knn_classify(test_instance, training_data, k)`
Classifies the test instance by finding the `k` nearest neighbors in the training data.

### 7. `evaluate_accuracy(test_data, training_data, k)`
Evaluates the accuracy of the k-NN model using the test dataset.

### 8. `leave_one_out_cross_validation(data, k)`
Performs leave-one-out cross-validation and calculates the accuracy of the k-NN model.

## How to Run

1. **Prepare your data**: Ensure you have the dataset in `dataset.json` and `test_dataset.json` format.

2. **Preprocess the data**: The data is preprocessed using one-hot encoding for categorical features and saved to the model file.

3. **Train the model**: Run the script to train the k-NN classifier on the dataset. It will evaluate the performance using leave-one-out cross-validation and print the confusion matrix and accuracy.

4. **Classify new instances**: You can classify new test instances by using the `knn_classify` function and specifying the value of `k`.

## Example Usage

```python
# Load the dataset
training_data = load_model("dataset.json")
test_data = load_model("test_dataset.json")

# Preprocess the data
one_hot_encoded_training_data = preprocess_data(training_data)
one_hot_encoded_test_data = preprocess_data(test_data)

# Save the preprocessed data
save_model(one_hot_encoded_training_data, "knn_model.json")

# Set k for k-NN
k = 3

# Evaluate model accuracy using leave-one-out cross-validation
accuracy = leave_one_out_cross_validation(one_hot_encoded_training_data, k)

print(f"Accuracy: {accuracy}")
```

## Results

The model produces the following confusion matrix during evaluation:

```
Confusion Matrix:
TP: 0, FP: 0
FN: 8, TN: 6
Accuracy: 0.42857142857142855
```

### Observations:
- The model's accuracy is 42.86%, with no true positives, indicating that it struggles to predict the positive class (`PlayTennis`).
- This suggests that the classifier may benefit from further optimization, such as adjusting the distance metric or dealing with imbalanced classes.

## Limitations

- The dataset may be imbalanced, which can lead to skewed performance.
- The model might be sensitive to the choice of features and the value of `k`.

## Future Work

- Experiment with different distance metrics (e.g., Minkowski distance, Cosine similarity).
- Handle class imbalance through techniques like oversampling, undersampling, or using weighted k-NN.
- Tune the hyperparameter `k` for better model performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```
