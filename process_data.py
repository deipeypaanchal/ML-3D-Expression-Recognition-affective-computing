'''
# Description: This file contains functions to process the data from the CK+ dataset.
# The get_class_label function returns the class label based on the input emotion label.
# The process_data function reads the data from the CK+ dataset and transforms the landmarks based on the input data type.
# It returns the features, classes, and subjects.
# The get_rotated_landmarks function rotates the landmarks around the x, y, and z axes.
'''

import os
import numpy as np
from landmarks import get_rotated_landmarks, get_translated_landmarks

def get_class_label(label):
    '''
    Returns the class label based on the input emotion label.

    Args:
        label (str): The emotion label.

    Returns:
        int: The corresponding class label.
    '''
    class_mapping = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5}
    return class_mapping.get(label)

def process_data(path, data_type):
    '''
    Reads the data from the CK+ dataset and transforms the landmarks based on the input data type.
    Returns the features, classes, and subjects.

    Args:
        path (str): The path to the CK+ dataset.
        data_type (str): The type of data transformation to apply.

    Returns:
        tuple: A tuple containing the features, classes, and subjects.
    '''
    feature_vectors = []   # List to store the features
    target_values = []    # List to store the classes
    subject_paths = []   # List to store the subjects
    
    print("Reading data...")
    # Loop through the dataset directory
    for subdir, dirs, files in os.walk(path):
        # Loop through the files in the subdirectory
        for file in files:
            # Check if the file is a .bnd file
            if file.endswith('.bnd'):
                # Get the label from the subdirectory
                emotion_label = subdir.split(os.path.sep)[-1]
                # Get the class label based on the emotion label
                label_value = get_class_label(emotion_label)
                # Get the path to the subject
                subject_filepath = os.path.join(subdir, file)
                # Read the landmarks from the file
                filepath = os.path.join(subdir, file)
                # Open the file and read the landmarks
                with open(filepath, 'r') as f:
                    # Initialize a list to store the landmarks
                    landmarks = []
                    # Read the landmarks from the file
                    bnd_file_contents = f.readlines()
                    # Loop through the landmarks and store them in the list
                    for curr_line in bnd_file_contents[:84]:
                        # Split the line and extract the x, y, and z coordinates
                        x, y, z = curr_line.split()[1:]
                        # Append the coordinates to the landmarks list
                        landmarks.append([float(x), float(y), float(z)])
                    # Check the data type and transform the landmarks accordingly
                    if data_type == "Original":
                        landmarks = np.array(landmarks).flatten()
                    elif data_type == "Translated":
                        landmarks = get_translated_landmarks(landmarks)
                    elif data_type == "Rotated":
                        landmarksRotatedX, landmarksRotatedY, landmarksRotatedZ = get_rotated_landmarks(landmarks)
                        landmarks = [landmarksRotatedX, landmarksRotatedY, landmarksRotatedZ]
                    # Append the landmarks, class label, and subject to the respective lists
                    feature_vectors.append(landmarks)
                    target_values.append(label_value)
                    subject_paths.append(subject_filepath)
    # Return the feature vectors, target values, and subject paths
    return feature_vectors, target_values, subject_paths