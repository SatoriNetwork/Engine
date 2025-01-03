
from satoriengine.veda.pipelines.chronos import ChronosVedaPipeline


def generate_training_data(ctx_len: int, num_features: int = 1, num_samples: int = 1000):
    """
    Generates synthetic training data for testing.
    Parameters:
    - ctx_len: The historical context length required for each sample (rows).
    - num_features: The number of features (columns) in the training data.
    - num_samples: The number of samples in the training set.
    Returns:
    - trainX: A list of pandas DataFrames, where each DataFrame represents one sample.
    - trainY: A numpy array of shape (num_samples, 1).
    - eval_set: A tuple containing evaluation X and Y datasets.
    """
    import pandas as pd
    import numpy as np
    # Generate synthetic training data
    trainX = []
    trainY = []
    for _ in range(num_samples):
        # Generate a sine wave with some noise
        data = np.sin(np.linspace(0, 20 * np.pi, ctx_len)) + \
            np.random.normal(0, 0.1, ctx_len)
        # Multiple features: stack the same time series with small variations
        features = np.stack([data + np.random.normal(0, 0.01, ctx_len)
                            for _ in range(num_features)], axis=1)
        # Target is the mean of the last 10 points
        target = np.mean(data[-10:]) + np.random.normal(0, 0.1)
        # Convert features to DataFrame
        df = pd.DataFrame(features, columns=[
                          f'feature_{i}' for i in range(num_features)])
        trainX.append(df)
        trainY.append(target)
    trainY = np.array(trainY)
    # Create evaluation set
    eval_size = max(10, int(num_samples * 0.1))
    evalX = trainX[:eval_size]
    evalY = trainY[:eval_size]
    eval_set = (evalX, evalY)
    return trainX, trainY, eval_set

def test():
    # Generate training data
    ctx_len = 512
    num_features = 3  # Example for multiple features
    trainX, trainY, eval_set = generate_training_data(
        ctx_len=ctx_len, num_features=num_features, num_samples=1000)
    # Pass the first sample as a DataFrame to predict
    c = ChronosVedaPipeline()
    # not deterministic
    c.predict(trainX[0])
    c.predict(trainX[0])
    c.predict(trainX[0])
    c.predict(trainX[0])
    c.predict(trainX[0])
    c.predict(trainX[0])
    c.predict(trainX[0])

def read_model():
    import joblib
    x = joblib.load('/Satori/Neuron/models/veda/kDlxC5Ayb7vN9sUVoTTUUvUPxiQ-/XgbChronosPipeline.joblib')
    df = x['dataset']
    return x, df
