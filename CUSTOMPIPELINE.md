# Custom Pipeline Implementation Guide

This guide explains how to create your own custom model pipeline by implementing the `PipelineInterface`. The interface provides a standardized way to integrate different machine learning models into the Engine.

## Table of Contents
- [Overview](#overview)
- [Interface Requirements](#interface-requirements)
- [Implementation Guide](#implementation-guide)
- [Testing Practices](#testing-practices)
- [Helper Data Processing Function](#helper-data-processing-function)

## Overview

The `PipelineInterface` defines a standard contract for model pipelines, ensuring compatibility with the prediction engine. Each pipeline must implement methods for:
- Model condition evaluation
- Model loading and saving
- Training and prediction
- Model comparison and scoring

## Interface Requirements

Your custom pipeline class must implement these methods from `PipelineInterface`:

### 1. `condition()`
```python
@staticmethod
def condition(*args, **kwargs) -> float:
```
Determines when the Engine should choose the pipeline for given conditions. Returns a value between 0 and 1:
- 0: Not suitable
- 1: Ideal conditions

Example criteria:
- Data size
- CPU/GPU availability
- Memory constraints
- Data characteristics

### 2. `load()` and `save()`
```python
def load(self, modelPath: str, **kwargs) -> Union[None, PipelineInterface]:
def save(self, modelpath: str, **kwargs) -> bool:
```
Handle model persistence:
- `load`: Restore model from disk
- `save`: Persist model to disk
- Use consistent serialization (e.g., joblib for sklearn models)

### 3. `fit()`
```python
def fit(self, *args, **kwargs) -> TrainingResult:
```
Trains the model and returns a `TrainingResult` object containing:
- Training status ( 1 - Success / other - Failure)
- Trained model instance
- Optional model error metrics

### 4. `compare()`
```python
def compare(self, *args, **kwargs) -> bool:
```
Compares model performance against a reference model:
- Usually compares against current stable model ( current best model )
- Returns True if new model performs better
- Should use consistent comparison metrics

### 5. `score()`
```python
def score(self, *args, **kwargs) -> float:
```
Calculates model performance metric:
- Returns a float value
- Lower values typically indicate better performance
- Should be consistent with comparison logic

### 6. `predict()`
```python
def predict(self, *args, **kwargs) -> Union[None, pd.DataFrame]:
```
Generates predictions:
- Returns predictions as a pandas DataFrame
- Should handle missing or invalid input gracefully
- Returns None if prediction fails

## Implementation Guide

1. Create a new class inheriting from `PipelineInterface`:
```python
from satoriengine.veda.pipelines.interface import PipelineInterface

class MyCustomPipeline(PipelineInterface):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = None
        self.modelError = None
        # Add custom initialization
```

2. Implement required methods:
```python
    @staticmethod
    def condition(*args, **kwargs) -> float:
    def load(self, modelPath: str, **kwargs) -> Union[None, PipelineInterface]:
    def save(self, modelpath: str, **kwargs) -> bool:
    def fit(self, *args, **kwargs) -> TrainingResult:
    def compare(self, *args, **kwargs) -> bool:
    def score(self, *args, **kwargs) -> float:
    def predict(self, *args, **kwargs) -> Union[None, pd.DataFrame]:
```

3. Implement Pipeline specific methods if any:
```python
    #Example ( Function for preparing time features for XGB Pipeline )
    @staticmethod
    def _prepareTimeFeatures(dates: np.ndarray) -> pd.DataFrame:
        """Convert datetime series into numeric features for XGBoost"""
        df = pd.DataFrame({'date_time': pd.to_datetime(dates)})
        df['hour'] = df['date_time'].dt.hour
        df['day'] = df['date_time'].dt.day
        df['month'] = df['date_time'].dt.month
        df['year'] = df['date_time'].dt.year
        df['day_of_week'] = df['date_time'].dt.dayofweek
        return df.drop('date_time', axis=1)
```

## Testing Practices

   - Test with various data scenarios
   - Validate model persistence
   - Check error handling

## Helper Data Processing Function

### Data Processing Made Easy
The library provides a helpful `process_data` function that handles most common data processing tasks:

```python
from satoriengine.veda.process import process_data

# Use in your pipeline, data being a dataframe with atleast 3 rows
processed = process_data(data)
```

What `process_data` does for you:
- Handles missing data automatically
- Divides datasets appropriately for training, validation and testing
- Creates useful dataset features
- Calculates sampling frequency of your data
- Determines number of forecasting steps, backtest steps.
- Provides other helpful dataset statistics ( check out [process.py](satoriengine/veda/process.py) to know more )

Example usage in a pipeline:
```python
def fit(self, data: pd.DataFrame, **kwargs):
    # Let process_data handle the heavy lifting
    processed = process_data(data)
    
    # Now you have access to:
    self.dataset = processed.dataset  # Processed dataset
    self.dataset_withfeatures = processed.dataset_withfeatures
    self.sampling_frequency = processed.sampling_frequency  # Data frequency
    # and others
    
    # Continue with your model training...
```

This saves you from having to write your own data preprocessing code and ensures consistency across different pipelines.

## Examples

For examples and reference implementations, check out [here](satoriengine/veda/pipelines).