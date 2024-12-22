import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from satoriengine.veda.pipelines.meta.transform import apply_transformation, fit_transformation

def generate_sample_dataset(num_samples=10, noise_level=0.1):
    """
    Generate a sample dataset with 2 features and a response variable.

    Parameters:
    - num_samples: Number of samples to generate (default is 100).
    - noise_level: Standard deviation of Gaussian noise to add to the response (default is 0.1).

    Returns:
    - A tuple (features, response) where:
        - features is a NumPy array (num_samples x 2).
        - response is a NumPy array (length num_samples).
    """
    np.random.seed(42)  # For reproducibility
    # Generate two random features
    feature_1 = np.random.uniform(-10, 10, num_samples)
    feature_2 = np.random.uniform(-10, 10, num_samples)
    # Define a response variable with some relationship to the features
    #response = 3 * feature_1 + 2 * feature_2 + np.random.normal(0, noise_level, num_samples)
    # Define one without an easily definable relationship
    response = np.random.normal(0, noise_level, num_samples)
    # Combine features into a single array
    features = np.column_stack((feature_1, feature_2))
    return features, response

def visualizeToPng(features):
    # Plot the 'value' column against the 'date_time' index
    plt.figure(figsize=(12, 6))
    plt.scatter(features[:, 0], features[:, 1], c=response, cmap='viridis')
    for i in range(len(features)):
        plt.text(features[i, 0], features[i, 1], str(i), fontsize=12, ha='right', va='top')
    # Formatting the plot
    plt.title("Time Series Visualization", fontsize=16)
    plt.xlabel("Date Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    # Save the plot to a file
    output_path = './visualize.png'
    plt.savefig(output_path)
    # Close the plot to free memory
    plt.close()
    print(f"Plot saved to {output_path}")


# Generate sample data
features, response = generate_sample_dataset()
visualizeToPng(features)
x = fit_transformation(features, response)
input('continue?')
newf = apply_transformation(x, features)
visualizeToPng(newf)



def view():
    import matplotlib.pyplot as plt
    # Plot the features and response
    plt.scatter(features[:, 0], features[:, 1], c=response, cmap='viridis')
    for i in range(len(features)):
        plt.text(features[i, 0], features[i, 1], str(i), fontsize=8, ha='center', va='center')
    plt.colorbar(label='Response Variable')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Sample Dataset with 2 Features')
    plt.show()
