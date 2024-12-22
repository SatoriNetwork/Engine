import numpy as np
from scipy.optimize import minimize_scalar


def random_point_on_circle(radius):
    """
    Returns a random point (x, y) uniformly distributed on the circumference
    of a circle centered at the origin with a given radius.
    """
    rng = np.random.default_rng(42)
    theta = 2 * np.pi * rng.random()
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def optimal_circle_placement(df, a_new, b_new, radius):
    # df has columns: x, y, a, b
    print(df)
    if len(df) == 1:
        return (1,0)
    if len(df) == 2:
        return (-1,0)
    if len(df) == 3:
        return (0,1)
    if len(df) == 4:
        return (0,-1)
    # Compute similarities (sum of absolute differences)
    similarities = np.abs(df['a'].values - a_new) + np.abs(df['b'].values - b_new)
    median_similarity = np.median(similarities)

    # Points with similarity <= median get their similarity as weight
    # Points with similarity > median get weight = 0
    weights = np.where(similarities <= median_similarity, similarities, 0)

    x_vals = df['x'].values
    y_vals = df['y'].values

    def objective(theta):
        x_new = radius * np.cos(theta)
        y_new = radius * np.sin(theta)
        distances = np.sqrt((x_vals - x_new)**2 + (y_vals - y_new)**2)
        return np.sum(weights * distances)

    res = minimize_scalar(objective, bounds=(0, 2*np.pi), method='bounded')
    theta_opt = res.x
    x_opt = radius * np.cos(theta_opt)
    y_opt = radius * np.sin(theta_opt)
    return x_opt, y_opt

def optimal_circle_placement_gravity(df, a_new, b_new, radius, epsilon=1e-9):
    # df has columns: x, y, a, b
    print(df)
    if len(df) == 1:
        return (1,0)
    if len(df) == 2:
        return (-1,0)
    if len(df) == 3:
        return (0,1)
    if len(df) == 4:
        return (0,-1)
    # Compute similarities
    similarities = np.abs(df['a'].values - a_new) + np.abs(df['b'].values - b_new)
    print(similarities)
    # Determine median similarity
    #median_similarity = np.median(similarities)
    # If similarity is zero, that would lead to infinite weight.
    # We'll handle that by adding a small epsilon.
    weights = 1.0 / ((similarities + epsilon)**2)
    print(weights)
    # Extract coordinates
    x_vals = df['x'].values
    y_vals = df['y'].values
    # Objective function: given theta, compute weighted sum of distances
    def objective(theta):
        x_new = radius * np.cos(theta)
        y_new = radius * np.sin(theta)
        distances = np.sqrt((x_vals - x_new)**2 + (y_vals - y_new)**2)
        return np.sum(weights * distances)
    # Minimize over theta in [0, 2*pi)
    res = minimize_scalar(objective, bounds=(0, 2*np.pi), method='bounded')
    # Compute final optimal point
    theta_opt = res.x
    x_opt = radius * np.cos(theta_opt)
    y_opt = radius * np.sin(theta_opt)
    return x_opt, y_opt


def optimal_circle_placement_antigravity(df, a_new, b_new, radius, epsilon=1e-3):
    # df has columns: x, y, a, b
    print(df)
    if len(df) == 1:
        return (1,0)
    if len(df) == 2:
        return (-1,0)
    if len(df) == 3:
        return (0,1)
    if len(df) == 4:
        return (0,-1)

    similarities = np.abs(df['a'].values - a_new) + np.abs(df['b'].values - b_new)

    median_s = np.median(similarities)
    def weight_func(s):
        diff = s - median_s
        # sign(diff): negative if s<median, positive if s>median, zero if s=median
        # weight(s) = -sign(diff) / (diff^2 + epsilon)
        # If s<median, diff<0 -> sign(diff)=-1 => weight=+1/(diff²+ε) (attractive)
        # If s>median, diff>0 -> sign(diff)=+1 => weight=-1/(diff²+ε) (repulsive)
        if diff == 0:
            return 0.0
        return -np.sign(diff) / (diff**2 + epsilon)

    weights = np.array([weight_func(s) for s in similarities])

    x_vals = df['x'].values
    y_vals = df['y'].values

    def objective(theta):
        x_new = radius * np.cos(theta)
        y_new = radius * np.sin(theta)
        distances = np.sqrt((x_vals - x_new)**2 + (y_vals - y_new)**2)
        return np.sum(weights * distances)

    res = minimize_scalar(objective, bounds=(0, 2*np.pi), method='bounded')
    theta_opt = res.x
    x_opt = radius * np.cos(theta_opt)
    y_opt = radius * np.sin(theta_opt)
    return x_opt, y_opt


import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def optimal_circle_placement_geometric(df, radius):
    # df has columns: x, y
    x_vals = df['x'].values
    y_vals = df['y'].values

    def objective(theta):
        x_new = radius * np.cos(theta)
        y_new = radius * np.sin(theta)
        distances = np.sqrt((x_vals - x_new)**2 + (y_vals - y_new)**2)
        return np.sum(distances)

    res = minimize_scalar(objective, bounds=(0, 2*np.pi), method='bounded')
    theta_opt = res.x
    x_opt = radius * np.cos(theta_opt)
    y_opt = radius * np.sin(theta_opt)
    return x_opt, y_opt
