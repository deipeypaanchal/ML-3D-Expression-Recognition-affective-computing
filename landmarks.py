'''
Description: This file contains the functions to translate and rotate the landmarks for the dataset.

The functions in this file perform the following operations:
- get_translated_landmarks: Translates the landmarks to the origin by subtracting the mean from each coordinate.
- get_rotated_landmarks: Rotates the landmarks around the x, y, and z axes using rotation matrices.
'''

import numpy as np
from math import acos

# Function to get the translated landmarks for the dataset
def get_translated_landmarks(landmarks):
    '''
    Translates the landmarks to the origin by subtracting the mean from each coordinate.

    Parameters:
    - landmarks: A list or array of landmarks with shape (N, 3), where N is the number of landmarks and each landmark has 3 coordinates (x, y, z).

    Returns:
    - translated_landmarks: A flattened numpy array of translated landmarks.

    '''
    # Convert the landmarks to a numpy array for easier manipulation and calculation
    landmarks = np.array(landmarks)
    # Calculate the mean of the x, y, and z coordinates
    x_mean = np.mean(landmarks[:, 0])
    y_mean = np.mean(landmarks[:, 1])
    z_mean = np.mean(landmarks[:, 2])
    # Translate the landmarks to the origin by subtracting the mean from each coordinate
    landmarks[:, 0] -= x_mean
    landmarks[:, 1] -= y_mean
    landmarks[:, 2] -= z_mean
    # Return the translated landmarks
    return landmarks.flatten()

# Function to get the rotated landmarks for the dataset
def get_rotated_landmarks(landmarks):
    '''
    Rotates the landmarks around the x, y, and z axes using rotation matrices.

    Parameters:
    - landmarks: A list or array of landmarks with shape (N, 3), where N is the number of landmarks and each landmark has 3 coordinates (x, y, z).

    Returns:
    - x_rotated: A flattened numpy array of landmarks rotated around the x-axis.
    - y_rotated: A flattened numpy array of landmarks rotated around the y-axis.
    - z_rotated: A flattened numpy array of landmarks rotated around the z-axis.

    '''
    # Convert the landmarks to a numpy array for easier manipulation and calculation
    landmarks = np.array(landmarks)
    # Calculate the value of pi
    pi = round(2 * acos(0.0), 3)
    # Calculate the cosine and sine of pi
    cos_pi = np.cos(pi)
    sin_pi = np.sin(pi)
    # Create the rotation matrices for each axis (x, y, z)
    x_axis = np.array([[1, 0, 0], [0, cos_pi, sin_pi], [0, -sin_pi, cos_pi]])
    y_axis = np.array([[cos_pi, 0, -sin_pi], [0, 1, 0], [sin_pi, 0, cos_pi]])
    z_axis = np.array([[cos_pi, sin_pi, 0], [-sin_pi, cos_pi, 0], [0, 0, 1]])
    # Rotate the landmarks for each axis
    x_rotated = x_axis.dot(landmarks.T).T
    y_rotated = y_axis.dot(landmarks.T).T
    z_rotated = z_axis.dot(landmarks.T).T
    # Return the rotated landmarks
    return x_rotated.flatten(), y_rotated.flatten(), z_rotated.flatten()