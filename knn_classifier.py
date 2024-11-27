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

# Calculate Euclidean distance
def calculate_distance(instance1, instance2):
    return np.sqrt(np.sum((np.array(instance1) - np.array(instance2)) ** 2))

def knn_classify(test_instance, training_data, k):
    distances = []

    # Extract the features of the test instance (exclude the label in test_instance)
    test_features = list(test_instance.values())[:-1]  # Exclude label from test instance

    for train_instance in training_data:
        # Extract the features of the training instance (exclude the label)
        train_features = list(train_instance.values())[:-1]  # Exclude label from training instance
        
        # Calculate Euclidean distance between test instance and training instance
        dist = calculate_distance(test_features, train_features)
        distances.append((dist, train_instance[-1]))  # Append the distance and the label of the train instance

    # Sort distances by the calculated distance
    distances.sort(key=lambda x: x[0])

    # Select the k nearest neighbors
    neighbors = distances[:k]

    # Perform majority voting to determine the predicted label
    labels = [neighbor[1] for neighbor in neighbors]
    prediction = max(set(labels), key=labels.count)

    return prediction, distances



def evaluate_accuracy(test_data, training_data, k):
    correct = 0
    with open("classification_log.txt", "w") as log_file:
        for instance in test_data:
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

# Main function to classify and evaluate
if __name__ == "__main__":
    # Example usage
    training_data = load_model("dataset.json")
    test_data = load_model("test_dataset.json")

    
    one_hot_encoded_training_data = preprocess_data(training_data)
    one_hot_encoded_test_data = preprocess_data(test_data)

    save_model(one_hot_encoded_training_data, "knn_model.json")

    k = int(input("Enter the value of k: "))


    knn_classify(one_hot_encoded_test_data, one_hot_encoded_training_data, k)
    #accuracy = evaluate_accuracy(one_hot_encoded_training_data, one_hot_encoded_training_data, k)

    print(f"Accuracy: {accuracy}")
   


