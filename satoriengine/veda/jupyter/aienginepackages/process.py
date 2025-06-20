import warnings
warnings.filterwarnings('ignore')
from typing import Union
from datetime import datetime, timedelta
from pandas.tseries.frequencies import to_offset
from scipy.stats import kruskal
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures



# from satorilib.logging import debug, info, error
from logging import debug


class ProcessedData:
    def __init__(
        self,
        end_times: dict,
        dataset: pd.DataFrame,
        data_subsets: dict,
        dataset_withfeatures: pd.DataFrame,
        dataset_with_features_subsets: dict,
        dataset_start_time: pd.Timestamp,
        dataset_end_time: pd.Timestamp,
        sampling_frequency: str,
        lags: int,
        backtest_steps: int,
        forecasting_steps: int,
        use_weight: bool,
        time_metric_baseline: str,
        forecasterequivalentdate: int,
        forecasterequivalentdate_n_offsets: int,
        if_invalid_dataset: bool,
        allowed_models: list,
    ):
        self.end_times = end_times
        self.dataset = dataset
        self.data_subsets = data_subsets
        self.dataset_withfeatures = dataset_withfeatures
        self.dataset_with_features_subsets = dataset_with_features_subsets
        self.dataset_start_time = dataset_start_time
        self.dataset_end_time = dataset_end_time
        self.sampling_frequency = sampling_frequency
        self.lags = lags
        self.backtest_steps = backtest_steps
        self.forecasting_steps = forecasting_steps
        self.use_weight = use_weight
        self.time_metric_baseline = time_metric_baseline
        self.forecasterequivalentdate = forecasterequivalentdate
        self.forecasterequivalentdate_n_offsets = forecasterequivalentdate_n_offsets
        self.if_invalid_dataset = if_invalid_dataset
        self.allowed_models = allowed_models


def fractional_hour_generator(datetimeparameter):
    # print('starting function')
    whole_time = datetimeparameter.time()
    fractional_hour = whole_time.hour + whole_time.minute / 60.0 + 1
    return fractional_hour


def test_seasonality(differenced_dataset, SF, sampling_frequency):
    debug(f"new seasonality test with SF = {SF}")

    if SF == "month":
        differenced_dataset[SF] = differenced_dataset.index.month
    elif SF == "week":
        differenced_dataset[SF] = differenced_dataset.index.isocalendar().week
    elif SF == "day_of_week":
        differenced_dataset[SF] = differenced_dataset.index.dayofweek + 1
    elif SF == "hour":
        differenced_dataset[SF] = differenced_dataset.index.hour + 1
    elif SF == "fractional_hour":
        differenced_dataset[SF] = differenced_dataset.index.map(
            fractional_hour_generator
        )
    else:
        # For any other potential SF values
        try:
            differenced_dataset[SF] = getattr(differenced_dataset.index, SF)
        except AttributeError:
            debug(f"Error: {SF} is not a valid attribute of the DatetimeIndex.")
            return False, None

    unique_seasonal_frequency = differenced_dataset[SF].unique()

    if len(unique_seasonal_frequency) < 2:
        debug(
            f"{SF.capitalize()} has less than 2 unique values. Cannot perform seasonality test."
        )
        return False, None

    res = []
    for i in unique_seasonal_frequency:
        group_data = differenced_dataset[differenced_dataset[SF] == i]["value"]
        if not group_data.empty:
            res.append(group_data)
        else:
            debug(f"Seasonal frequency {i} has no data.")

    if len(res) < 2:
        debug(
            f"{SF.capitalize()} has less than 2 non-empty groups. Cannot perform seasonality test!!!!."
        )
        return False, None
    try:
        H_statistic, p_value = kruskal(*res)
        p_value = round(p_value, 3)
        seasonal = p_value <= 0.05

        debug(f"{SF.capitalize()} H_statistic is {H_statistic}")
        debug(f"{SF.capitalize()} p_value is {p_value}")
        debug(f"Seasonality that is built on {SF} is {seasonal}")

        return seasonal, p_value
    except ValueError as e:
        debug(f"Error in seasonality test for {SF}: {str(e)}")
        return False, None


def create_exogenous_features(
    original_dataset,
    optimally_differenced_dataset,
    dataset_start_time,
    dataset_end_time,
    include_fractional_hour=False,
    exogenous_feature_type="AdditiveandMultiplicativeExogenousFeatures",
    sampling_frequency="h",
):
    if exogenous_feature_type == "NoExogenousFeatures":
        return [], pd.DataFrame(), pd.DataFrame(), False, False, False

    # Initialize new dataset
    new_dataset = pd.DataFrame(index=original_dataset.index)
    dataset_start_time = dataset_start_time.strftime("%Y-%m-%d %H:%M:%S")
    dataset_end_time = dataset_end_time.strftime("%Y-%m-%d %H:%M:%S")

    start_date_time_object = datetime.strptime(dataset_start_time, "%Y-%m-%d %H:%M:%S")
    end_date_time_object = datetime.strptime(dataset_end_time, "%Y-%m-%d %H:%M:%S")

    dataset_delta = end_date_time_object - start_date_time_object

    dataset_offset = to_offset("{td.days}D{td.seconds}s".format(td=dataset_delta))

    SeasonalFrequency = []

    sampling_frequency_offset = to_offset(sampling_frequency)
    hour_test = to_offset("1h")
    day_of_week_test = to_offset("1d")
    week_test = to_offset("7d")
    month_test = to_offset("31d")
    year_test = to_offset("366d")

    if sampling_frequency_offset <= hour_test and dataset_offset >= to_offset("3d"):
        SeasonalFrequency.append("hour")
        if sampling_frequency_offset < hour_test and (
            (
                exogenous_feature_type
                in [
                    "ExogenousFeaturesBasedonSeasonalityTest",
                    "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative",
                ]
            )
            or include_fractional_hour == True
        ):
            SeasonalFrequency.append("fractional_hour")

    if sampling_frequency_offset <= day_of_week_test and dataset_offset >= to_offset(
        "21d"
    ):
        SeasonalFrequency.append("day_of_week")

    if sampling_frequency_offset <= week_test and dataset_offset >= to_offset("1095d"):
        SeasonalFrequency.append("week")

    if sampling_frequency_offset <= month_test and dataset_offset >= to_offset("1095d"):
        SeasonalFrequency.append("month")

    # in the future we can add in holidays and yearly_quarters
    if sampling_frequency_offset <= year_test and dataset_offset >= to_offset("1095d"):
        SeasonalFrequency.append("year")

    # Create all calendar features
    for SF in SeasonalFrequency:
        if SF == "hour":
            # we should consider for odd sampling freq may decide for the value to be a fraction ( need different formula )
            new_dataset[SF] = new_dataset.index.hour + 1
        elif SF == "fractional_hour":
            new_dataset[SF] = new_dataset.index.map(
                fractional_hour_generator
            )  # set the right parameter
        elif SF == "day_of_week":
            new_dataset[SF] = new_dataset.index.dayofweek + 1
        elif SF == "week":
            new_dataset[SF] = new_dataset.index.isocalendar().week
        elif SF == "month":
            new_dataset[SF] = new_dataset.index.month
        elif SF == "year":
            new_dataset[SF] = new_dataset.index.year
    # Create cyclical features
    for feature in new_dataset.columns:
        new_dataset[f"sin_{feature}"] = np.sin(
            2 * np.pi * new_dataset[feature] / new_dataset[feature].max()
        )
        new_dataset[f"cos_{feature}"] = np.cos(
            2 * np.pi * new_dataset[feature] / new_dataset[feature].max()
        )
    frac_hour_seasonal = None
    hour_seasonal = None
    day_of_week_seasonal = None
    week_seasonal = None
    if exogenous_feature_type in [
        "ExogenousFeaturesBasedonSeasonalityTest",
        "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative",
    ]:
        # Perform seasonality tests
        # run the seasonality test on both hour and fractional_hour to test both
        # if none is seasonal ( seasonality test is false for both )  then both return false for seasonality
        # if one is true then we return for that seasonality as true and the other as false
        # if both are true for seasonality then we want to return to one as true and the other as false ( priority for smaller p_value and return its seasonality as true ) (if p_value same then return hour seasonality as true)
        seasonal_periods = []
        p_values = {}
        chosen_hour_type = None  # This will store either 'hour' or 'fractional_hour'

        # Special handling for hour and fractional_hour
        if "hour" in SeasonalFrequency:
            hour_seasonal, hour_p_value = test_seasonality(
                optimally_differenced_dataset, "hour", sampling_frequency
            )
            frac_hour_seasonal, frac_hour_p_value = test_seasonality(
                optimally_differenced_dataset, "fractional_hour", sampling_frequency
            )
        if "day_of_week" in SeasonalFrequency:
            day_of_week_seasonal, day_of_week_p_value = test_seasonality(
                optimally_differenced_dataset, "day_of_week", sampling_frequency
            )
        if "week" in SeasonalFrequency:
            week_seasonal, week_p_value = test_seasonality(
                optimally_differenced_dataset, "week", sampling_frequency
            )
            # calculation of week seasonality test to determine seasonal period of a year can be improved in the future by using day_of_year instead of week_of_year
        if not hour_seasonal and not frac_hour_seasonal:
            debug("Neither hour nor fractional_hour is seasonal.")
        elif hour_seasonal and frac_hour_seasonal:
            if hour_p_value <= frac_hour_p_value:
                chosen_hour_type = "hour"
                seasonal_periods.append("hour")
                p_values["hour"] = hour_p_value
                debug(
                    "Both hour and fractional_hour are seasonal. Choosing hour due to lower or equal p-value."
                )
            else:
                chosen_hour_type = "fractional_hour"
                seasonal_periods.append("fractional_hour")
                p_values["fractional_hour"] = frac_hour_p_value
                debug(
                    "Both hour and fractional_hour are seasonal. Choosing fractional_hour due to lower p-value."
                )
        elif hour_seasonal:
            chosen_hour_type = "hour"
            seasonal_periods.append("hour")
            p_values["hour"] = hour_p_value
            debug("Only hour is seasonal.")
        elif frac_hour_seasonal:
            chosen_hour_type = "fractional_hour"
            seasonal_periods.append("fractional_hour")
            p_values["fractional_hour"] = frac_hour_p_value
            debug("Only fractional_hour is seasonal.")

        # Test other seasonal frequencies
        for SF in [
            sf for sf in SeasonalFrequency if sf not in ["hour", "fractional_hour"]
        ]:
            is_seasonal, p_value = test_seasonality(
                optimally_differenced_dataset, SF, sampling_frequency
            )
            if is_seasonal:
                seasonal_periods.append(SF)
                p_values[SF] = p_value

        if not seasonal_periods:
            debug(
                "No seasonal periods detected. Returning no exogenous calendar related features."
            )
            new_dataset = pd.DataFrame(index=original_dataset.index)
        else:
            debug("Detected seasonal periods:")
            for period in seasonal_periods:
                debug(f"{period}: p-value = {p_values[period]}")

            # Keep only features for seasonal periods, excluding the non-chosen hour type
            seasonal_features = [
                col
                for col in new_dataset.columns
                if any(period in col for period in seasonal_periods)
                and (
                    chosen_hour_type not in ["hour", "fractional_hour"]
                    or (chosen_hour_type == "hour" and "fractional_hour" not in col)
                    or (chosen_hour_type == "fractional_hour" and "hour" in col)
                )
            ]
            new_dataset = new_dataset[seasonal_features]

    num_columns = new_dataset.shape[1]

    if (
        exogenous_feature_type
        in [
            "AdditiveandMultiplicativeExogenousFeatures",
            "ExogenousFeaturesBasedonSeasonalityTestWithAdditivenMultiplicative",
        ]
        and num_columns > 0
    ):
        # Apply polynomial features (multiplicative case)
        polynomialobject2 = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False  # was False
        ).set_output(transform="pandas")

        num_columns = new_dataset.shape[1]
        new_dataset = polynomialobject2.fit_transform(new_dataset.dropna())

    # Get exogenous feature names
    exog_features = new_dataset.columns.tolist()

    # Create final dataframe of exogenous features
    df_exogenous_features = new_dataset.copy()
    return (
        exog_features,
        new_dataset,
        df_exogenous_features,
        hour_seasonal,
        day_of_week_seasonal,
        week_seasonal,
    )


def generate_exog_data(end_date, freq, steps, date_format):
    end_validation = pd.to_datetime(end_date, format=date_format)

    # Generate date range for the exogenous series
    date_range = pd.date_range(
        start=end_validation + pd.Timedelta(freq), periods=steps, freq=freq
    )

    # Create exog_series with 0 values
    exog_series = pd.Series(0, index=date_range)

    # Create exog_timewindow
    exog_timewindow = exog_series.reset_index()
    exog_timewindow.columns = ["date_time", "value"]
    exog_timewindow["date_time"] = pd.to_datetime(
        exog_timewindow["date_time"], format=date_format
    )
    exog_timewindow = exog_timewindow.set_index("date_time")
    exog_timewindow = exog_timewindow.asfreq(freq)

    return exog_series, exog_timewindow


def filter_dataframe_col(df, selected_col):
    col_to_keep = [col for col in df.columns if col in selected_col]
    return df[col_to_keep]


def roundto_minute(roundingdatetime):
    """
    Round the given timedelta to the nearest minute and return the minute component.
    """
    # Add 30 seconds for rounding
    rounded = roundingdatetime + timedelta(seconds=30)
    total_minutes = rounded.total_seconds() / 60
    minutes = int(total_minutes % 60)
    return minutes


def round_time(
    first_day,
    dt,
    round_to_hours,
    round_to_minutes,
    round_to_seconds,
    offset_hours=0,
    offset_minutes=0,
    offset_seconds=0,
):
    # Apply offset
    dt = dt + timedelta(
        hours=offset_hours, minutes=offset_minutes, seconds=offset_seconds
    )

    # Calculate total seconds for the rounding interval
    total_seconds = (round_to_hours * 3600) + (round_to_minutes * 60) + round_to_seconds

    # Calculate seconds since first day
    seconds_since_first = round((dt - first_day).total_seconds())

    # Round to the nearest interval
    rounded_seconds = round(seconds_since_first / total_seconds) * total_seconds

    # Create new datetime with rounded seconds
    rounded_dt = first_day + timedelta(seconds=rounded_seconds)

    # Remove the offset
    rounded_dt -= timedelta(
        hours=offset_hours, minutes=offset_minutes, seconds=offset_seconds
    )

    return rounded_dt


def round_to_nearest_minute(dt):
    return dt.replace(second=0, microsecond=0) + timedelta(
        minutes=1 if dt.second >= 30 else 0
    )


def process_noisy_dataset(
    df,
    round_to_hours,
    round_to_minutes,
    round_to_seconds=0,
    offset_hours=0,
    offset_minutes=0,
    offset_seconds=0,
    datetime_column="date_time",
):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Check if the datetime is in a column or is the index
    if isinstance(df_copy.index, pd.DatetimeIndex):
        datetime_series = df_copy.index
        is_index = True
    elif datetime_column in df_copy.columns:
        datetime_series = pd.to_datetime(df_copy[datetime_column])
        is_index = False
    else:
        raise ValueError(f"Datetime column '{datetime_column}' not found in DataFrame")

    # Get the first day and round it to the nearest minute
    first_day = round_to_nearest_minute(datetime_series.min())

    # Apply round_time function
    rounded_datetimes = datetime_series.map(
        lambda x: round_time(
            first_day,
            x,
            round_to_hours,
            round_to_minutes,
            round_to_seconds,
            offset_hours,
            offset_minutes,
            offset_seconds,
        )
    )

    # Update the DataFrame
    if is_index:
        df_copy.index = rounded_datetimes
    else:
        df_copy[datetime_column] = rounded_datetimes

    return df_copy


def process_data(
    data: pd.DataFrame,
    sampling_frequency: Union[str, None] = None,
    col_names: Union[list[str], None] = None,
    training_percentage: int = 80,
    validation_percentage: int = 10,
    # test_percentage: int = 10,
    quick_start: bool = False,
) -> ProcessedData:

    # if col_names is None:
    #     col_names = ["date_time", "value", "id"]

    # Read the CSV file
    raw_dataset = data

    # Process date_time column with flexible parsing and standardization
    raw_dataset["date_time"] = pd.to_datetime(raw_dataset["date_time"], utc=True)

    # Convert to '%Y-%m-%d %H:%M:%S' format
    raw_dataset["date_time"] = raw_dataset["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    raw_dataset["date_time"] = pd.to_datetime(
        raw_dataset["date_time"], format="%Y-%m-%d %H:%M:%S"
    )

    raw_dataset = raw_dataset.set_index("date_time")

    raw_diff_dat = raw_dataset.index.to_series().diff()
    value_counts = raw_diff_dat.value_counts().sort_index()
    num_distinct_values = len(value_counts)

    if num_distinct_values > (len(raw_dataset) * 0.05):
        median = raw_dataset.index.to_series().diff().median()

        if median < timedelta(hours=1, minutes=0, seconds=29):
            if median >= timedelta(minutes=59, seconds=29):
                round_to_hour = 1
                round_to_minute = 0
            else:
                round_to_hour = 0
                round_to_minute = roundto_minute(median)
        else:
            round_to_hour = median.total_seconds() // 3600
            round_to_minute = roundto_minute(
                median - timedelta(hours=round_to_hour, minutes=0, seconds=0)
            )

        dataset = process_noisy_dataset(
            raw_dataset, round_to_hours=round_to_hour, round_to_minutes=round_to_minute
        )

    else:
        dataset = raw_dataset

    if sampling_frequency is None:
        sf = dataset.index.to_series().diff().median()

        # Convert to frequency string
        abbr = {
            "days": "d",
            "hours": "h",
            "minutes": "min",
            "seconds": "s",
            "milliseconds": "ms",
            "microseconds": "us",
            "nanoseconds": "ns",
        }

        def fmt(sf):
            return "".join(
                f"{v}{abbr[k]}" for k, v in sf.components._asdict().items() if v != 0
            )

        if isinstance(sf, pd.Timedelta):
            sampling_frequency = fmt(sf)
        elif isinstance(sf, pd.TimedeltaIndex):
            sampling_frequency = sf.map(fmt)
        else:
            raise ValueError

    # Convert sampling_frequency to timedelta for comparison
    # sf_timedelta = pd.Timedelta(sampling_frequency)

    # # Check if sampling frequency is between 1 day and 1.5 days
    # if pd.Timedelta(days=1) < sf_timedelta < pd.Timedelta(days=1.5):
    #     sampling_frequency = '1d'  # Set to exactly one day

    # Handle duplicates
    duplicates = dataset[dataset.index.duplicated(keep=False)]
    duplicates_sorted = duplicates.sort_index()

    dataset_averaged = dataset.groupby(level=0).agg(
        {
            "value": "mean",
            "id": "first",  # Keep the first ID, or you could use 'last' or another method
        }
    )

    # Replace the original dataset with the averaged one
    dataset = dataset_averaged

    # maybe should be changed
    dataset = dataset.asfreq(sampling_frequency, method="nearest")

    datasetsize = len(dataset)

    training_index = round(training_percentage / 100 * datasetsize)
    validation_index = min(
        training_index + round(validation_percentage / 100 * datasetsize),
        datasetsize - 2,
    )
    test_index = datasetsize - 1

    dataset = dataset.reset_index()
    end_train = dataset.iloc[training_index]["date_time"]
    end_validation = dataset.iloc[validation_index]["date_time"]
    end_test = dataset.iloc[test_index]["date_time"]

    dataset_start_time = dataset.iloc[0]["date_time"]

    dataset = dataset.set_index("date_time")

    dataset = dataset.asfreq(sampling_frequency)

    # Split data into subsets
    data_train = dataset.loc[:end_train, :]
    data_val = dataset.loc[end_train:end_validation, :].iloc[1:]
    data_trainandval = dataset.loc[:end_validation, :]
    data_test = dataset.loc[end_validation:end_test, :].iloc[1:]

    # Prepare return values
    end_times = {"train": end_train, "validation": end_validation, "test": end_test}

    data_subsets = {
        "train": data_train,
        "validation": data_val,
        "train_and_val": data_trainandval,
        "test": data_test,
    }

    dataset_end_time = dataset.index[-1]

    _, _, df_exogenous_features, _, _, _ = create_exogenous_features(
        original_dataset=dataset,
        optimally_differenced_dataset=dataset,
        dataset_start_time=dataset_start_time,
        dataset_end_time=dataset_end_time,
        include_fractional_hour=True,
        exogenous_feature_type="AdditiveandMultiplicativeExogenousFeatures",
        sampling_frequency=sampling_frequency,
    )

    dataset_withfeatures = dataset[["value"]].merge(
        df_exogenous_features, left_index=True, right_index=True, how="left"
    )

    dataset_withfeatures = dataset_withfeatures.astype(
        {
            col: np.float32
            for col in dataset_withfeatures.select_dtypes("number").columns
        }
    )
    data_train_withfeatures = dataset_withfeatures.loc[:end_train, :].copy()
    data_val_withfeatures = dataset_withfeatures.loc[end_train:end_validation, :].copy()
    data_test_withfeatures = dataset_withfeatures.loc[end_validation:, :].copy()

    dataset_with_features_subsets = {
        "train": data_train_withfeatures,
        "validation": data_val_withfeatures,
        "test": data_test_withfeatures,
    }

    all_models = [
        "baseline", # has three models inside it
        "direct_linearregression",
        "direct_ridge",
        "direct_lasso",
        "direct_linearboost",
        "direct_lightgbm",
        "autoreg_linearregression",
        "autoreg_ridge",
        "autoreg_lasso",
        "autoreg_linearboost",
        "autoreg_lightgbm",
        "autoreg_histgradient",
        "autoreg_xgb",
        "autoreg_catboost",
        "arima",
        "skt_prophet_additive",
        "skt_prophet_hyper",
        "skt_ets",
        "skt_tbats_damped",
        "skt_tbats_standard",
        "skt_tbats_quick",
        "skt_lstm_deeplearning",
        "autoreg_randomforest",
        "direct_xgb",
        "direct_catboost",
        "direct_histgradient",
    ]

    allowed_models = [
        "baseline",
        "direct_linearregression",
        # "direct_ridge",
        "direct_lasso",
        "direct_linearboost",
        "autoreg_linearregression",
        # "autoreg_ridge",
        "autoreg_lasso",
        "autoreg_linearboost",
        "autoreg_lightgbm",
        "autoreg_histgradient",
        "autoreg_xgb",
        "autoreg_catboost",
        "arima",
        "skt_ets",
        "skt_tbats_damped",
        "skt_prophet_additive",
        "skt_prophet_hyper",
        "skt_tbats_standard",
        "skt_tbats_quick",
        "skt_lstm_deeplearning",
    ]

    # for testing purposes
    # allowed_models = ['direct_lightgbm']

    dataset_duration = dataset_end_time - dataset_start_time
    sampling_timedelta = pd.Timedelta(sampling_frequency)
    week_timedelta = pd.Timedelta(days=7)
    steps_in_week = int(week_timedelta / sampling_timedelta)

    if sampling_timedelta > pd.Timedelta(hours=1):
        lags = round(min(len(data_subsets["test"]), steps_in_week))
    else:
        lags = round(min(0.3 * len(dataset), steps_in_week))

    if_invalid_dataset = False
    time_metric_baseline = "hours"
    forecasterequivalentdate = 1
    forecasterequivalentdate_n_offsets = 1

    # Remove Prophet models if dataset duration is less than 2 years
    if dataset_duration < pd.Timedelta(days=365 * 2):
        allowed_models = [
            model
            for model in allowed_models
            if model not in ["skt_prophet_additive", "skt_prophet_hyper"]
        ]

    forecasting_steps = None
    sampling_timedelta = pd.Timedelta(sampling_frequency)
    week_timedelta = pd.Timedelta(days=7)
    day_timedelta = pd.Timedelta(days=1)

    if sampling_timedelta < day_timedelta:
        # Calculate steps relative to a day
        forecasting_steps = min(
            int(day_timedelta / sampling_timedelta), len(data_subsets["test"])
        )
    elif sampling_timedelta >= day_timedelta:
        # Calculate steps relative to a week
        # can override the setting but have to not do backtesting
        forecasting_steps = min(
            int(week_timedelta / sampling_timedelta), len(data_subsets["test"])
        )

    if dataset_duration >= pd.Timedelta(days=19) and len(dataset) >= 25:
        # debug("Hits the >= 25 length dataset case and >= 19 days")
        # quick_start : linear_regression with no_exog, feature_set_reduction = False
        if quick_start:
            allowed_models = ["autoreg_lightgbm"]

        use_weight = True
        time_metric_baseline = "days"
        forecasterequivalentdate = 1
        forecasterequivalentdate_n_offsets = 7
    else:
        # quick_start : linear_regression with no_exog, feature_set_reduction = False
        if dataset_duration >= pd.Timedelta(days=3) and len(dataset) >= 72:
            # debug("Hits the >= 72 length dataset case and >= 3 days")
            allowed_models = [
                "baseline",
                "direct_linearregression",
                # "direct_ridge",
                "direct_lasso",
                "direct_xgb",
                "direct_catboost",
                "direct_histgradient",
                "autoreg_linearregression",
                # "autoreg_ridge",
                "autoreg_lasso",
                "autoreg_lightgbm",
                "autoreg_histgradient",
                "autoreg_xgb",
                "autoreg_catboost",
                "arima",
                "skt_ets",
            ]
            if quick_start:
                allowed_models = ["autoreg_lightgbm"]
            time_metric_baseline = "days"
            forecasterequivalentdate = 1
            forecasterequivalentdate_n_offsets = min(dataset_duration.days - 1, 1)

        elif dataset_duration >= pd.Timedelta(hours=2) and len(dataset) >= 12:  # was if dataset_duration >= pd.Timedelta(hours=12) and len(dataset) >= 12
            # quick_start : baseline with no_exog, feature_set_reduction = False
            allowed_models = [
                "baseline",
                "direct_linearregression",
                "autoreg_lightgbm",
            ]
            if quick_start:
                allowed_models = ["autoreg_lightgbm"]
            # debug("Hits the >= 12 length dataset case and >= 12 hours")

            if sampling_timedelta > pd.Timedelta(hours=1):
                time_metric_baseline = "days"
                forecasterequivalentdate_n_offsets = min(dataset_duration.days - 1, 1)
            else:
                time_metric_baseline = "hours"
                forecasterequivalentdate_n_offsets = int(
                    dataset_duration.total_seconds() / 7200
                )

            forecasterequivalentdate = 1

        elif len(dataset) >= 3:
            allowed_models = ["autoreg_lightgbm", "direct_linearregression"]
            if dataset_duration >= pd.Timedelta(hours=2):
                allowed_models.append("baseline")

            if quick_start:
                allowed_models = ["autoreg_lightgbm"]
            lags = 1
            forecasting_steps = 1
            if sampling_timedelta > pd.Timedelta(hours=1):
                time_metric_baseline = "days"
            else:
                time_metric_baseline = "hours"
            forecasterequivalentdate = 1
            forecasterequivalentdate_n_offsets = 1
        else:
            # debug("Hits the invalid dataset case")
            if_invalid_dataset = True

        use_weight = False

    if sampling_timedelta < pd.Timedelta(hours=4):
        allowed_models = [
            model
            for model in allowed_models
            if model not in ["direct_linearregression", "direct_xgb", "direct_catboost", "direct_histgradient", "arima",
                             "skt_lstm_deeplearning", "skt_tbats_damped", "skt_tbats_standard", "autoreg_histgradient",
                              "autoreg_xgb", "autoreg_catboost", "skt_ets", "skt_tbats_quick" ]
        ]

    backtest_steps = forecasting_steps

    nan_percentage = dataset.isna().mean()
    if nan_percentage.value > 0.4:
        use_weight = False

    return ProcessedData(
        end_times,
        dataset,
        data_subsets,
        dataset_withfeatures,
        dataset_with_features_subsets,
        dataset_start_time,
        dataset_end_time,
        sampling_frequency,
        int(lags),
        backtest_steps,
        forecasting_steps,
        use_weight,
        time_metric_baseline,
        forecasterequivalentdate,
        forecasterequivalentdate_n_offsets,
        if_invalid_dataset,
        allowed_models,
    )
