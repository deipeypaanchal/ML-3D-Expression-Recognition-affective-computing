"""
# Description: This file contains the function to evaluate the classifier using the cross-validation method.
# The evaluate function takes the features, classes, classifier, data type, and subjects as input.
# It initializes the classifier based on the input classifier type.
# It then calls the cross_validation function from classify.py to get the confusion matrix, precision, recall, and accuracy scores.
# It calculates the average confusion matrix, precision, recall, and accuracy scores.
# It creates a new directory to store the results if it does not exist.
# It creates a new file to store the results if it does not exist.
# It writes the results to the file in the new directory.
# It handles exceptions if there is an error writing to the file and prints the error message.
"""

import os
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from classify import cross_validation
from plot import generate_plot

# This function evaluates the classifier using the cross-validation method
def evaluate(X, y, classifier, data_type, subjects):
    """
    Evaluate the classifier using the cross-validation method.

    Args:
        X (array-like): The input features.
        y (array-like): The input classes.
        classifier (str): The type of classifier to use. Possible values: "SVM", "RF", "TREE".
        data_type (str): The type of data being evaluated.
        subjects (int): The number of subjects in the data.

    Returns:
        None

    Raises:
        None
    """
    # Initialize classifier
    clf = None
    # Check the classifier type
    if classifier == "SVM":
        # Use the linear support vector machine
        clf = svm.LinearSVC(dual=False)
    # Check the classifier type
    elif classifier == "RF": 
        # Use the random forest classifier
        clf = RandomForestClassifier()
    # Check the classifier type
    elif classifier == "TREE":
        # Use the decision tree classifier
        clf = DecisionTreeClassifier()
    print(f"Evaluating {classifier} classifier using {data_type} data...")
    # Here we are calling the cross_validation function from classify.py to get the confusion matrix, precision, recall, and accuracy scores
    confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores = cross_validation(X, y, clf, classifier, data_type, subjects)
    # Calculate the average confusion matrix, precision, recall, and accuracy scores
    print("Calculating average scores...")
    confusion_matrix_avg = np.mean((confusion_matrix_scores), axis=0)
    precision_avg = np.mean((precision_scores), axis=0)
    recall_avg = np.mean((recall_scores), axis=0)
    accuracy_avg = np.mean((accuracy_scores), axis=0)
    # Create a new directory to store the results if it does not exist
    new_directory = 'Results/' + classifier
    # Create a new file to store the results if it does not exist
    os.makedirs(new_directory, exist_ok=True)
    # Define the filename to store the results in the new directory
    filename = classifier + "_" + data_type + ".txt"
    # Write the results to the file in the new directory
    try:
        print("Writing the average scores to the file...")
        # Open the file in append mode so that the results are not overwritten
        with open(os.path.join(new_directory, filename), 'a') as file:
            # Write the average scores to the file
            file.write("---------------------------------------------\n")
            file.write("AVERAGE SCORES\n")
            file.write("---------------------------------------------\n")
            file.write(f'Confusion matrix: {confusion_matrix_avg}\n')
            file.write(f'Precision: {precision_avg}\n')
            file.write(f'Recall: {recall_avg}\n')
            file.write(f'Accuracy: {accuracy_avg}\n')
    # Handle the exception if there is an error writing to the file and print the error message
    except Exception as e:
        print("Error writing the file:", e)
    
    # Plot the data in a 3D scatter plot
    generate_plot(X, data_type)