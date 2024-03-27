'''
# Description: This file contains the function to plot the data in a 3D scatter plot using matplotlib.
# The plot function takes the data and label as input and plots the data in a 3D scatter plot.
# It assigns a different color to each class based on the label.
# It saves the plot as an image file with the label as the filename.
'''

import matplotlib.pyplot as plt
import random

def generate_plot(data, label):
    """
    Plots a 3D scatter plot of the given data with a specified label.

    Parameters:
    - data (numpy.ndarray): The data to be plotted. It should be a 2D array with shape (n, 3), where n is the number of samples.
    - label (str): The label to be assigned to the plotted data.

    Returns:
    None
    """
    print(f"Plotting data for {label}...")
    # Select a random sample from the data
    sample_index = random.randint(0, len(data))
    # Reshape the sample data so that we can take the needed columns and not the index
    data = data[sample_index].reshape(83, 3)
    # Create a 3D scatter plot
    fig = plt.figure()
    # Add a subplot with 3D projection
    ax = fig.add_subplot(111, projection='3d')
    # Define colors for each class
    colors = {'Original': 'r', 'Translated': 'g', 'RotatedX': 'b', 'RotatedY': 'y', 'RotatedZ': 'm'}
    c = colors.get(label, 'k')  # Default to black if label is not found in colors dictionary
    # Plot the data
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], label=label, color=c)
    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set the title of the plot
    ax.legend()
    # Save the plot as an image file with the label as the filename
    # plt.show()
    
    # save the plot as an image file with the label as the filename in Results directory
    plt.savefig(f"Results/Plots/{label}")

    print(f"Plot saved as {label}.png in Results directory")