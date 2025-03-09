import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame


def createTrainTest(data, forecasting_steps: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    data['date_time'] = pd.to_datetime(data['date_time'])  #convert to datetime
    timeseriesid = 1
    data['timeseriesid'] = timeseriesid #add new colum 'id' to dataframe
    data = TimeSeriesDataFrame.from_data_frame(
        data,
        id_column="timeseriesid",
        timestamp_column="date_time"
    )
    num_rows = data.num_timesteps_per_item()[1]
    data_train, data_traintest = data.train_test_split(
        prediction_length = forecasting_steps
    )
    return data_train, data_traintest

def conformData(target_df: pd.DataFrame, covariate_dfs: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    target_df = target_df.copy()
    target_df['date_time'] = pd.to_datetime(target_df['date_time'])
    target_df = target_df.set_index('date_time')
    start_time = target_df.index.min()
    end_time = target_df.index.max()
    sf = getSamplingFreq(target_df)
    full_range = pd.date_range(start=start_time, end=end_time, freq=sf)
    target_df = target_df.reindex(full_range).ffill()
    result_df = pd.DataFrame({'value': target_df['value']})
    result_df = result_df.reset_index().rename(columns={'index': 'date_time'})
    for i, cov_df in enumerate(covariate_dfs, 1):
        cov_df = cov_df.copy()
        cov_df['date_time'] = pd.to_datetime(cov_df['date_time'])
        cov_df = cov_df.set_index('date_time').sort_index()
        aligned_cov = cov_df.reindex(
            pd.DatetimeIndex(sorted(set(cov_df.index) | set(full_range))),
            method='ffill'
        )
        aligned_cov = aligned_cov.reindex(full_range)
        cov_col_name = f'covariate_{i}_value'
        result_df[cov_col_name] = aligned_cov['value'].values
    covariateColNames = [col for col in result_df.columns if col not in ['date_time', 'value']]
    return result_df, covariateColNames

def getSamplingFreq(dataset: pd.DataFrame) -> str:
    def fmt(sf):
        return "".join(
            f"{v}{abbr[k]}"
            for k, v in sf.components._asdict().items()
            if v != 0 and k in ["days", "hours", "minutes"])
        
    sf = dataset.index.to_series().diff().median()
    abbr = {
        "days": "d",
        "hours": "h",
        "minutes": "min",
        "seconds": "s",
        "milliseconds": "ms",
        "microseconds": "us",
        "nanoseconds": "ns"}
    if isinstance(sf, pd.Timedelta):
        sampling_frequency = fmt(sf)
    elif isinstance(sf, pd.TimedeltaIndex):
        sampling_frequency = sf.map(fmt)
    else:
        raise ValueError
    return sampling_frequency        