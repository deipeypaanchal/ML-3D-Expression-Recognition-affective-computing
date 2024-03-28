# 3D Expression Recognition Project

This project leverages machine learning to analyze 3D facial landmarks, employing three classic classifiers: Random Forest (RF), Support Vector Machine (SVM), and Decision Tree (TREE). Our aim is to assess the effectiveness of these classifiers across different data manipulations—original, translated, and rotated 3D landmarks—to generate insightful reports on their performance metrics.

## Packages Used

To ensure smooth execution of this project, the following Python packages are utilized:

- `sys`: For accessing system-specific parameters and functions.
- `os`: To interact with the operating system.
- `numpy`: Essential for numerical computing.
- `math`: Provides access to mathematical functions.
- `sklearn`: The core machine learning library used for implementing classifiers and evaluation metrics.
  - `sklearn.ensemble`: For ensemble-based machine learning algorithms.
  - `sklearn.tree`: Contains decision tree-based algorithms.
  - `sklearn.metrics`: Offers metrics to evaluate machine learning models.
  - `sklearn.model_selection`: Tools for model selection and evaluation.
- `random`: Generates random numbers.
- `matplotlib`: For plotting and visualizing data.

## Installation

Before running the project, ensure all required packages are installed. This can be done via pip:

```bash
pip install numpy scikit-learn matplotlib
```

For other packages (sys, os, math, and random), these come pre-installed with Python.

## Usage

To run the project, navigate to the project directory and execute the following command in your terminal:

```bash
python Project1.py <classifier> <data type> <data directory>
```

- `<classifier>`: Choose between RF, SVM, or TREE.
- `<data type>`: Specify the type of data manipulation (Original, Translated, or Rotated).
- `<data directory>`: The path to the data directory containing 3D facial landmarks.

Example:

```bash
python Project1.py TREE Original ./BU4DFE_BND_V1.1
```

## Project Structure

Ensure your data directory (`./BU4DFE_BND_V1.1`) is structured properly with subject directories and expression-specific `.BND` files as described.

## Experiments

The project conducts three main experiments with 10-fold cross-validation:

1. **Raw 3D Landmarks:** Utilize the original 3D landmarks as the feature vector.
2. **Translated 3D Landmarks:** Translate the landmarks to the origin by averaging and subtracting the coordinates.
3. **Rotated 3D Landmarks:** Rotate the landmarks 180 degrees over the x, y, and z axes.

For each classifier and experiment, calculate the confusion matrix, accuracy, precision, and recall. Remember to save these results for analysis.

## Note

It's recommended to install the libraries in a virtual environment to avoid conflicts with other projects or system-wide packages.