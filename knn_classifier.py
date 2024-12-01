import json
import numpy as np
import pandas as pd

# Convert categorical to numeric using one-hot encoding
def preprocess_data(dataset):
    df = pd.DataFrame(dataset)
    one_hot_encoded_df = pd.get_dummies(df, columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])
    return one_hot_encoded_df

# Save preprocessed data to JSON
def save_model(data, filename="knn_model.json"):
   # Convert the DataFrame to a dictionary that is serializable to JSON
    data_dict = data.to_dict(orient='records')
    
    # Save the data to a JSON file
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4)
    

# Update model with new training data
def update_model(new_data, filename="knn_model.json"):
    try:
        with open(filename, "r") as f:
            existing_data = json.load(f)
        existing_data.extend(new_data)
    except FileNotFoundError:
        existing_data = new_data

    save_model(existing_data, filename)

# Load the stored model
def load_model(filename):
    with open(filename, "r") as f:
        return json.load(f)

def calculate_distance(instance1, instance2):
    # Ensure both instances have the same columns
    if not instance1.columns.equals(instance2.columns):
        raise ValueError("Instances must have the same columns for distance calculation.")

    # Compute the squared differences for all features (exclude 'PlayTennis' column)
    squared_differences = [
        (int(instance1[column].values[0]) - int(instance2[column].values[0])) ** 2
        for column in instance1.columns if column != "PlayTennis"
    ]
    # Calculate the square root of the sum of squared differences
    return np.sqrt(sum(squared_differences))


# Calculate Manhattan Distance
def calculate_manhattan_distance(instance1, instance2):
    # Ensure both instances have the same columns
    if not instance1.columns.equals(instance2.columns):
        raise ValueError("Instances must have the same columns for distance calculation.")
    
    # Compute the absolute differences for all features (exclude 'PlayTennis' column)
    absolute_differences = [
        abs(int(instance1[column].values[0]) - int(instance2[column].values[0]))
        for column in instance1.columns if column != "PlayTennis"
    ]
    # Return the sum of the absolute differences
    return sum(absolute_differences)

def knn_classify(test_instance, training_data, k, distance_calculation_type):
    distances = []

    for i in range(len(training_data)):
        # Extract the current training instance as a DataFrame
        train_instance = training_data.iloc[i:i+1]

        # Calculate the distance between the test and training instance
        if distance_calculation_type == "E":
            dist = calculate_distance(test_instance, train_instance)
        elif distance_calculation_type == "M":
            dist = calculate_manhattan_distance(test_instance, train_instance)
        # Extract the label ('PlayTennis') as a scalar value
        label = train_instance['PlayTennis'].values[0]

        # Append the distance and label as a tuple
        distances.append((dist, label))

    # Sort distances by the calculated distance
    distances.sort(key=lambda x: x[0])

    # Select the k nearest neighbors
    neighbors = distances[:k]
    # Perform majority voting to determine the predicted label
    labels = [neighbor[1] for neighbor in neighbors]

    # Extract scalar value from each label if it's a Series
    labels = [label if isinstance(label, (int, str)) else label.values[0] for label in labels]

    prediction = max(set(labels), key=labels.count)
    
    return prediction, distances


def evaluate_accuracy(test_data, training_data, k):
    correct = 0
    with open("classification_log.txt", "w") as log_file:
        for i in range(len(test_data)):
            instance = training_data.iloc[i:i+1]
            # Ensure instance is a dictionary (not a string)
            if isinstance(instance, dict):
                # Extracting features (excluding label)
                features = list(instance.values())[:-1]  # Exclude the label
                actual = instance['PlayTennis']  # Assuming 'PlayTennis' is the label field
                
                # Classify the instance using the KNN classifier
                prediction, _ = knn_classify(features, training_data, k)
                
                # Write the prediction and actual class to the log file
                log_file.write(f"Predicted: {prediction}, Actual: {actual}\n")
                
                # Check if the prediction matches the actual label
                if prediction == actual:
                    correct += 1
            else:
                # Handle the case where the instance is not a dictionary
                log_file.write(f"Invalid data format: {instance}\n")
    
    # Return accuracy as the ratio of correct predictions
    return correct / len(test_data)

def leave_one_out_cross_validation(data, k, distance_calculation_type):
    # Initialize confusion matrix counters
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    correct_predictions = 0

    for i in range(len(data)):
        # Use the ith instance as the test set
        test_instance = data.iloc[i:i+1]

        # Use all other instances as the training set
        training_data = pd.concat([data.iloc[:i], data.iloc[i+1:]])

        # Extract the actual label of the test instance
        actual_label = test_instance["PlayTennis"].values[0]

        # Classify the test instance using the KNN classifier
        prediction, _ = knn_classify(test_instance, training_data, k, distance_calculation_type)

        # Check if the prediction matches the actual label
        if prediction == actual_label:
            correct_predictions += 1
            if prediction == 1:  # True Positive
                true_positive += 1
            else:  # True Negative
                true_negative += 1
        else:
            if prediction == 1:  # False Positive
                false_positive += 1
            else:  # False Negative
                false_negative += 1

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(f"TP: {true_positive}, FP: {false_positive}")
    print(f"FN: {false_negative}, TN: {true_negative}")

    # Calculate and return the average accuracy
    accuracy = correct_predictions / len(data)
    return accuracy

# Main function to classify and evaluate
if __name__ == "__main__":
    # Example usage
    training_data = load_model("dataset.json")
    test_data = load_model("test_dataset.json")
    
    training_data_df = pd.DataFrame(training_data)
    print("Training Dataset:")
    print(training_data_df.to_string(index=False)) 

    # Summarize the dataset by counting instances per class (i.e., PlayTennis)
    class_counts = training_data_df['PlayTennis'].value_counts()
    print("\nClass Distribution (PlayTennis):")
    print(class_counts)
    
    one_hot_encoded_training_data = preprocess_data(training_data)
    one_hot_encoded_test_data = preprocess_data(test_data)

    save_model(one_hot_encoded_training_data, "knn_model.json")

    k = int(input("Enter the value of k: "))

    distance_calculation_type  = str(input("Enter the type of distance calculation method E for Euclidean - M for Manhattan: "))

    accu_loocv = leave_one_out_cross_validation(one_hot_encoded_training_data,k,distance_calculation_type)

    print(f"Accuracy: {accu_loocv}")