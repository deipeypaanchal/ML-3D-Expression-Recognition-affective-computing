'''
# Description: This file contains the functions to perform cross-validation and print evaluation metrics
# The PrintEvalMetrics function prints the evaluation metrics to a file
# The cross_validation function performs cross-validation using GroupKFold
# It returns the confusion matrix, precision, recall, and accuracy scores
# It writes the results to a file in the Results directory
# It handles exceptions if there is an error writing to the file and prints the error message
'''

import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GroupKFold

# Print evaluation metrics to a file
def PrintEvalMetrics(pred, indices, y, classifier, data_type, fold_index):
    """
    Prints evaluation metrics such as confusion matrix, precision, recall, and accuracy.
    
    Parameters:
    - pred (list): List of predicted labels.
    - indices (list): List of indices for the ground truth labels.
    - y (array-like): Ground truth labels.
    - classifier (str): Name of the classifier.
    - data_type (str): Type of data.
    - fold_index (int): Index of the fold.
    
    Returns:
    - cm (array): Confusion matrix.
    - precision (float): Precision score.
    - recall (float): Recall score.
    - accuracy (float): Accuracy score.
    """
    final_predictions = []
    ground_truth = []
    # Flatten the list of predicted labels
    for p in pred:
        # Extend the final_predictions list with the predicted labels
        final_predictions.extend(p)
    # Flatten the list of indices for the ground truth labels
    for i in indices:
        # Extend the ground_truth list with the ground truth labels
        ground_truth.extend(y[i])
    # Calculate the confusion matrix, precision, recall, and accuracy scores
    cm = confusion_matrix(final_predictions, ground_truth)
    # Calculate the precision score
    precision = precision_score(ground_truth, final_predictions, average='macro')
    # Calculate the recall score
    recall = recall_score(ground_truth, final_predictions, average='macro')
    # Calculate the accuracy score
    accuracy = accuracy_score(ground_truth, final_predictions)
    # Create a new directory to store the results if it does not exist
    new_dir = 'Results/' + classifier
    # Create a new file to store the results if it does not exist
    os.makedirs(new_dir, exist_ok=True)
    # Define the filename to store the results in the new directory
    filename = classifier + "_" + data_type + ".txt"
    # Write the results to the file in the new directory
    try:
        # Open the file in append mode
        with open(os.path.join(new_dir, filename), 'a') as file:
            file.write('Fold ' + str(fold_index)+'\n')              # Write the fold index to the file
            file.write("\tConfusion matrix: "+str(cm) + '\n')       # Write the confusion matrix to the file
            file.write("\tPrecision: " + str(precision) + '\n')     # Write the precision score to the file
            file.write("\tRecall: " + str(recall) + '\n')           # Write the recall score to the file
            file.write("\tAccuracy: " + str(accuracy) + '\n')       # Write the accuracy score to the file
    # Handle the exception if there is an error writing to the file and print the error message
    except Exception as e:
        # Print the error message
        print("Error while writing to file:", e)
    # Return the confusion matrix, precision, recall, and accuracy scores
    return cm, precision, recall, accuracy

# Perform cross-validation using GroupKFold
def cross_validation(X, y, clf, classifier, data_type, subject_ids):
    """
    Performs cross-validation on the given data using the specified classifier.
    
    Parameters:
    - X (array-like): Input features.
    - y (array-like): Target labels.
    - clf (object): Classifier object with fit and predict methods.
    - classifier (str): Name of the classifier.
    - data_type (str): Type of data.
    - subjects (array-like): Group labels for subjects.
    
    Returns:
    - confusion_matrix_scores (list): List of confusion matrices for each fold.
    - precision_scores (list): List of precision scores for each fold.
    - recall_scores (list): List of recall scores for each fold.
    - accuracy_scores (list): List of accuracy scores for each fold.
    """
    # Set the number of folds
    N_folds = 10
    # Set the groups for GroupKFold
    data_groups = subject_ids
    # Initialize GroupKFold
    group_k_fold = GroupKFold(n_splits=N_folds)
    # Get the number of splits
    group_k_fold.get_n_splits(X, y, data_groups)
    # Initialize lists to store evaluation metrics
    confusion_matrix_scores = []
    # Initialize lists to store evaluation metrics
    precision_scores = []
    # Initialize lists to store evaluation metrics
    recall_scores = []
    # Initialize lists to store evaluation metrics
    accuracy_scores = []
    # Loop through the folds
    print("Performing cross-validation...")
    for i in range(10):
        # Loop through the splits
        print(f'Writing the results to the file... {i + 1}/10')
        for fold_index, (train_index, test_index) in enumerate(group_k_fold.split(X, y, data_groups)):
            # Check if the fold index matches the current fold
            if i == fold_index:
                # Split the data into training and testing sets
                train_x, X_test = X[train_index], X[test_index]
                train_y, y_test = y[train_index], y[test_index]
                # Fit the classifier on the training data
                clf.fit(train_x, train_y)
                # Predict the labels on the testing data
                pred = clf.predict(X_test)
                # Print the evaluation metrics
                cm, precision, recall, accuracy = PrintEvalMetrics([pred], [test_index], y, classifier, data_type, fold_index + 1)
                # Append the evaluation metrics to the respective lists
                confusion_matrix_scores.append(cm)
                # Append the evaluation metrics to the respective lists
                precision_scores.append(precision)
                # Append the evaluation metrics to the respective lists
                recall_scores.append(recall)
                # Append the evaluation metrics to the respective lists
                accuracy_scores.append(accuracy)
                # Get the new groups for the next fold
                training_subjects = data_groups[train_index]
                # Get the new groups for the next fold
                test_data_groups = data_groups[test_index]
                # Get the new groups for the next fold
                fold_subjects = set(test_data_groups) - set(training_subjects)
                # Get the new groups for the next fold
                new_index = [i for i, subject in enumerate(data_groups) if subject in fold_subjects]
                # Concatenate the new groups with the training data
                train_x = np.concatenate((train_x, X[new_index]))
                # Concatenate the new groups with the training data
                train_y = np.concatenate((train_y, y[new_index]))
    # Return the confusion matrix, precision, recall, and accuracy scores
    return confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores