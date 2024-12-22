from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def fit_transformation(features, response, n_components=None):
    """
    Fit a transformation based on the relationship between features and response.

    Parameters:
    - features: A NumPy array or Pandas DataFrame (m x n).
    - response: A NumPy array or Pandas Series (length m).
    - n_components: Number of components for the transformation (default is min(n_features, m_samples)).

    Returns:
    - A dictionary containing:
        - 'scaler_features': Fitted StandardScaler for features.
        - 'scaler_response': Fitted StandardScaler for response.
        - 'pls_model': Fitted PLSRegression model.
    """
    # Normalize features and response
    scaler_features = StandardScaler()
    scaler_response = StandardScaler()

    X_scaled = scaler_features.fit_transform(features)
    y_scaled = scaler_response.fit_transform(response.reshape(-1, 1)).flatten()

    # Fit PLS model
    if n_components is None:
        n_components = min(features.shape[1], features.shape[0])
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X_scaled, y_scaled)

    # Return the transformation components
    return {
        'scaler_features': scaler_features,
        'scaler_response': scaler_response,
        'pls_model': pls_model
    }

def apply_transformation(transformation, features):
    """
    Apply a fitted transformation to a new dataset.

    Parameters:
    - transformation: A dictionary containing the fitted transformation (from fit_transformation).
    - features: A NumPy array or Pandas DataFrame (m x n).

    Returns:
    - Transformed features (m x n_components).
    """
    scaler_features = transformation['scaler_features']
    pls_model = transformation['pls_model']

    # Normalize the features
    X_scaled = scaler_features.transform(features)

    # Apply the transformation
    X_transformed = pls_model.transform(X_scaled)

    return X_transformed

# Functions are ready to be used.
# Let me know if you want a demo with a sample dataset like Iris!
