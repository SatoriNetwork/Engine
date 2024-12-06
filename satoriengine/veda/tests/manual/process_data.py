import numpy as np
import pandas as pd
from satoriengine.veda.process import process_data
def getData():
    return pd.DataFrame({
    'date_time': [
        '2024-11-26 15:48:58.072345',
        '2024-11-26 15:58:30.443838',
        '2024-11-26 16:06:15.397004',
        '2024-11-26 16:17:05.289652',
        '2024-11-26 16:57:03.294215',
        '2024-11-26 17:07:08.680330',
        '2024-11-26 17:37:44.601735',
        '2024-11-26 17:47:58.276935',
        '2024-11-26 18:09:48.342047',
        '2024-11-26 18:16:02.024987',
        '2024-11-26 18:37:46.504852',
        '2024-11-26 19:17:55.053249',
        '2024-11-26 19:27:43.275411',
        '2024-11-26 19:47:37.518203',
        '2024-11-26 19:58:43.648003',
        '2024-11-26 20:09:21.245749',
        '2024-11-26 20:17:04.835939',
        '2024-11-26 20:29:16.098032',
        '2024-11-26 20:34:39.778499',
        '2024-11-26 20:48:56.756648',
        '2024-11-26 20:59:04.937680',
        '2024-11-26 21:10:10.636831',
        '2024-11-26 21:59:17.267148',
        '2024-11-26 22:04:02.562980',
        '2024-11-26 22:17:43.444319',
        '2024-11-26 22:27:55.291734',
        '2024-11-26 22:36:43.316949',
        '2024-11-26 22:58:08.074115',
        '2024-11-26 23:04:55.062204',
        '2024-11-26 23:17:23.094097',
        '2024-11-26 23:27:21.992645',
        '2024-11-26 23:37:19.024551',
        '2024-11-26 23:47:16.076293',
        '2024-11-26 23:57:13.999362',
        '2024-11-27 00:07:27.089241',
        '2024-12-01 03:27:17.861137',
        '2024-12-01 03:37:23.397228',
        '2024-12-01 03:47:22.456126',
        '2024-12-01 03:57:34.846724',
        '2024-12-01 04:17:21.731181',
        '2024-12-01 04:27:08.613308',
        '2024-12-01 04:37:12.989472',
        '2024-12-01 04:47:16.910692',
        '2024-12-01 04:57:08.350604',
        '2024-12-01 05:07:19.648944',
        '2024-12-03 17:17:24.149210',
        '2024-12-03 17:27:50.748823',
        '2024-12-03 17:37:36.418936',
        '2024-12-03 17:47:59.625469',
        '2024-12-03 17:57:45.430551',
        '2024-12-04 05:08:09.207628',
        '2024-12-04 19:38:29.249589',
    ], 'value': [
        547.945205,
        544.959128,
        547.945205,
        547.945205,
        534.759358,
        540.540541,
        539.083558,
        539.083558,
        539.083558,
        537.634409,
        542.005420,
        544.959128,
        546.448087,
        549.450549,
        550.964187,
        549.450549,
        547.945205,
        547.945205,
        547.945205,
        549.450549,
        546.448087,
        547.945205,
        543.478261,
        544.959128,
        544.959128,
        546.448087,
        546.448087,
        539.083558,
        543.478261,
        542.005420,
        543.478261,
        540.540541,
        539.083558,
        542.005420,
        544.959128,
        458.715596,
        460.829493,
        461.893764,
        461.893764,
        462.962963,
        460.829493,
        460.829493,
        461.893764,
        462.962963,
        461.893764,
        472.813239,
        470.588235,
        472.813239,
        470.588235,
        464.037123,
        378.787879,
        339.558574,
    ], 'id': [
        '5a70e79a2ee7eacc',
        '0e272e0c2939702c',
        '7f722cd1ee0a0dc5',
        '72829c81bbcfc714',
        'a23fa30092fc723e',
        '5a74f601e772aa9c',
        'c955efa0798cc9aa',
        'c5aba73db44086f6',
        '11f5e7a6b4f554f5',
        '0e83eba69b30990a',
        '1c8f07aaec089639',
        '74a5ab5393356955',
        'ad6b88a5bb2c6430',
        '951a5beccdde048a',
        '20dbd6d1c1dc0d85',
        'a6bb404915d24d05',
        '3bb2db99aae91ee9',
        '714cfc08a9d73502',
        '832a242e4567593e',
        '346dfbcde38f8ed6',
        'fafebb8e1993c7f9',
        '19af3b4d810af3d2',
        '006ff53e376e6b3f',
        '123f73af1f2dbdd9',
        '86f581a1163e76bc',
        '415fac82141a59a4',
        'ef9ce6571358e906',
        '915623d64482852b',
        'a3adae94260cdb95',
        'c7b0f92456baae5f',
        '52d19a97c4e51ac6',
        '717152b7933890ee',
        '974765f25505289d',
        '65925ce2382d2e36',
        '030a30fc16475155',
        'ed93087dbb1910c4',
        '6ca2437c1bc72e0f',
        '2f7f62702154633c',
        'dc36d4d5613bf462',
        '74df49032490b44c',
        '4a3dde62217b7dea',
        '64c3a3a249d3cce5',
        'f876577a49e38518',
        '2f42d825e73535de',
        '626743c9b71d20eb',
        'c4424ed9b0c99882',
        '7d1b8f40cb122ca0',
        '8c556d88db50fda2',
        'fcda05601b8495ef',
        '457ccdbb115a04fb',
        'c592894a30cf42b4',
        '1c139da9a2fafd14',
    ]})
    #data['date_time'] = pd.to_datetime(data['date_time'])

def test_process_data():
    return process_data(getData())


def visualizeLocally(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    # Plot the 'value' column against the 'date_time' index
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['value'], label='Value', linewidth=1)
    # Formatting the plot
    plt.title("Time Series Visualization", fontsize=16)
    plt.xlabel("Date Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    # Show the plot
    plt.show()


def visualizeToPng(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    # Plot the 'value' column against the 'date_time' index
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['value'], label='Value', linewidth=1)
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
    output_path = '/Satori/Engine/satoriengine/veda/tests/manual/visualize.png'
    plt.savefig(output_path)
    # Close the plot to free memory
    plt.close()
    print(f"Plot saved to {output_path}")


def straight_line_interpolation(df, value_col, step='10T'):
    """
    Performs fractal interpolation on missing timestamps.

    Parameters:
    - df: DataFrame with a datetime index and a column to interpolate.
    - value_col: The column name with values to interpolate.
    - step: The frequency to use for resampling (e.g., '10T' for 10 minutes).

    Returns:
    - DataFrame with interpolated values.
    """
    # Ensure the DataFrame has a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'])
            df.set_index('date_time', inplace=True)
        else:
            raise ValueError("The DataFrame must have a DatetimeIndex or a 'date_time' column.")
    # Sort the index and resample
    df = df.sort_index()
    df = df.resample(step).mean()  # Resample to fill in missing timestamps with NaN
    # Perform fractal interpolation
    for _ in range(5):  # Number of fractal iterations
        filled = df[value_col].interpolate(method='linear')  # Linear interpolation
        perturbation = np.random.normal(scale=0.01, size=len(filled))  # Small random noise
        df[value_col] = filled + perturbation  # Add fractal-like noise
    return df


def merge(dfs: list[pd.DataFrame], targetColumn: 'str|tuple[str]'):
    ''' Layer 1
    combines multiple mutlicolumned dataframes.
    to support disparate frequencies,
    outter join fills in missing values with previous value.
    filters down to the target column observations.
    '''
    from functools import reduce
    if len(dfs) == 0:
        return None
    if len(dfs) == 1:
        return dfs[0]
    for ix, item in enumerate(dfs):
        if targetColumn in item.columns:
            dfs.insert(0, dfs.pop(ix))
            break
        # if we get through this loop without hitting the if
        # we could possibly use that as a trigger to use the
        # other merge function, also if targetColumn is None
        # why would we make a dataset without target though?
    for df in dfs:
        df.index = pd.to_datetime(df.index)
    return reduce(
        lambda left, right:
            pd.merge_asof(left, right, left_index=True, right_index=True),
        dfs)

fdf = fractal_stepwise_interpolation(df, 'value'); visualizeToPng(fdf)


import numpy as np
import pandas as pd
from scipy.stats import entropy

def entropy_signature(data, bins=20, segments=10):
    """
    Computes the entropy signature of a dataset by analyzing its variability across segments.
    Parameters:
    - data: 1D array-like or pandas Series (numeric values).
    - bins: Number of bins for calculating probabilities (default: 20).
    - segments: Number of segments to divide the data into for local entropy analysis (default: 10).
    Returns:
    - global_entropy: Overall Shannon entropy of the entire dataset.
    - segment_entropies: List of entropy values for each segment.
    - signature: Array of segment entropies normalized by global entropy.
    """
    # Ensure the data is a numpy array
    data = np.asarray(data.dropna() if isinstance(data, pd.Series) else data)
    # Compute the global entropy
    hist, _ = np.histogram(data, bins=bins, density=True)
    global_entropy = entropy(hist + 1e-9)  # Add small constant to avoid log(0)
    # Divide the data into segments
    segment_size = len(data) // segments
    segment_entropies = []
    for i in range(segments):
        segment = data[i * segment_size:(i + 1) * segment_size]
        if len(segment) > 0:
            hist, _ = np.histogram(segment, bins=bins, density=True)
            segment_entropy = entropy(hist + 1e-9)  # Add small constant
            segment_entropies.append(segment_entropy)
        else:
            segment_entropies.append(0.0)
    # Normalize segment entropies by the global entropy
    signature = np.array(segment_entropies) / (global_entropy + 1e-9)
    return {
        "global_entropy": global_entropy,
        "segment_entropies": segment_entropies,
        "signature": signature,
    }


def identify_islands_and_gaps(df, value_col, threshold_factor=3):
    """
    Identifies islands of real data and the gaps between them based on typical timestamp intervals.
    Parameters:
    - df: DataFrame with a datetime index and a column containing real data (with NaNs for missing values).
    - value_col: The column name with values to check for islands.
    - threshold_factor: Factor of the median interval to determine a "large gap" (default: 3).
    Returns:
    - islands: List of tuples, each containing the start and end indices of a contiguous island.
    - gaps: List of tuples, each containing the start and end indices of gaps between islands.
    """
    # Ensure the DataFrame is sorted by index
    df = df.sort_index()
    # Calculate time differences between consecutive timestamps
    time_deltas = df.index.to_series().diff().dt.total_seconds()
    typical_interval = time_deltas.median()  # Use the median interval as the typical interval
    large_gap_threshold = typical_interval * threshold_factor  # Threshold for detecting large gaps
    # Identify where data is present (islands) and missing (gaps)
    islands = []
    gaps = []
    start = None
    for i in range(len(df)):
        if not pd.isna(df[value_col].iloc[i]):  # Data is present
            if start is None:  # Start of a new island
                start = df.index[i]
            if i == len(df) - 1:  # Handle the case where the last row is part of an island
                islands.append((start, df.index[i]))
        else:  # Data is missing
            if start is not None:  # End of an island
                islands.append((start, df.index[i - 1]))
                start = None
    # Identify gaps based on large time differences
    for i in range(1, len(df)):
        if pd.isna(df[value_col].iloc[i]) and pd.isna(df[value_col].iloc[i - 1]):
            delta = (df.index[i] - df.index[i - 1]).total_seconds()
            if delta > large_gap_threshold:
                gaps.append((df.index[i - 1], df.index[i]))
    return {
        "islands": islands,
        "gaps": gaps,
        "typical_interval": typical_interval,
        "large_gap_threshold": large_gap_threshold,
    }


def combined_entropy_signature(df, value_col, bins=20, segments=10, threshold_factor=3):
    """
    Computes the combined entropy signature of a dataset by averaging the entropy signatures
    of islands of contiguous data.
    Parameters:
    - df: DataFrame with a datetime index and a column containing real data (with NaNs for missing values).
    - value_col: The column name with values to check for islands.
    - bins: Number of bins for calculating probabilities in entropy calculation (default: 20).
    - segments: Number of segments to divide each island for local entropy analysis (default: 10).
    - threshold_factor: Factor of the median interval to determine a "large gap" (default: 3).
    Returns:
    - combined_signature: A dictionary containing:
        - 'global_entropy': Combined global entropy of all islands.
        - 'segment_entropies': Averaged segment entropies across all islands.
        - 'signature': Combined normalized entropy signature.
    """
    # Step 1: Identify islands of contiguous data
    gaps_and_islands = identify_islands_and_gaps(df, value_col, threshold_factor)
    islands = gaps_and_islands["islands"]
    if not islands:
        return {
            "global_entropy": 0.0,
            "segment_entropies": [],
            "signature": np.zeros(segments),
        }
    # Step 2: Compute entropy signatures for each island
    global_entropies = []
    segment_entropies_list = []
    signatures_list = []
    for start, end in islands:
        # Extract island data
        island_data = df.loc[start:end, value_col].dropna()
        # Compute entropy signature for this island
        island_signature = entropy_signature(island_data, bins=bins, segments=segments)
        # Collect results
        global_entropies.append(island_signature["global_entropy"])
        segment_entropies_list.append(island_signature["segment_entropies"])
        signatures_list.append(island_signature["signature"])
    # Step 3: Combine entropy signatures
    combined_global_entropy = np.mean(global_entropies)
    combined_segment_entropies = np.mean(segment_entropies_list, axis=0)
    combined_signature = np.mean(signatures_list, axis=0)
    return {
        "global_entropy": combined_global_entropy,
        "segment_entropies": combined_segment_entropies.tolist(),
        "signature": combined_signature.tolist(),
    }
