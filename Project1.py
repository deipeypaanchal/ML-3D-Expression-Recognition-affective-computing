# Name: Deipey Paanchal
# U-Number: U80305771

'''
# Description: This script is the main entry point for the project. It reads the command line arguments and calls the process_data and evaluate functions.
# The main function checks if the classifier and data type are valid before calling the process_data function to read the data.
# It then calls the evaluate function to perform cross-validation and print the evaluation metrics.
# The main function takes the classifier, data type, and data directory as command line arguments.
# Usage: python Project1.py <classifier> <data_type> <data_directory>
# Example: python Project1.py SVM Original ./BU4DFE_BND_V1.1
'''

import sys
import numpy as np
from evaluate import evaluate
from process_data import process_data

def main():
    """
    ----------------------------------------------------------------------------------------------
    Main function for executing the project.
    ----------------------------------------------------------------------------------------------
    This function takes command line arguments for the classifier, data type, and data directory.
    It validates the inputs and processes the data accordingly.
    The processed data is then evaluated using the specified classifier and data type.
    """
    if len(sys.argv) != 4:
        print("Usage: python Project1.py <classifier> <data_type> <data_directory>")
        print("Example: python Project1.py TREE Original ./BU4DFE_BND_V1.1")
        print("Please make sure that you have the libraries needed for this project installed globally otherwise use the virtual environment.")
        print("If using virtual env make sure to activate it first and then: venv/bin/python Project1.py <classifier> <data_type> <data_directory>")
        sys.exit()
    
    # Read command line arguments
    classifier = sys.argv[1]
    data_type = sys.argv[2]
    data_directory = sys.argv[3]
    # Validate classifier and data type
    valid_classifiers = ["SVM", "RF", "TREE"]
    valid_data_types = ["Original", "Translated", "Rotated"]
    # Check if the classifier and data type are valid 
    if (classifier not in valid_classifiers) or (data_type not in valid_data_types):
        print("Invalid classifier or data type")
        print("Usage: python Project1.py <classifier> <data_type> <data_directory>")
        sys.exit()
    print("Process started...")        
    print(f"{classifier} {data_type}")
    # Process the data
    X, y, sub = process_data(data_directory, data_type)
    # Convert the data to numpy arrays
    X, y, sub = np.array(X), np.array(y), np.array(sub)
    # Evaluate the data based on the data type
    if data_type == "Original" or data_type == "Translated":
        evaluate(X, y, classifier, data_type, sub)
    elif data_type == "Rotated":
        # Split the data into x, y, and z coordinates
        rotated_x, rotated_y, rotated_z = X[:, 0], X[:, 1], X[:, 2]
        # Evaluate the data for each coordinate
        evaluate(rotated_x, y, classifier, data_type + "X", sub)
        evaluate(rotated_y, y, classifier, data_type + "Y", sub)
        evaluate(rotated_z, y, classifier, data_type + "Z", sub)

    print("Completed!")

# Call the main function
if __name__ == "__main__": 
    main()