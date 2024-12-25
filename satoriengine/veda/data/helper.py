# file for data related functions
import pandas as pd
from typing import Any


def validate_dataframe(df: pd.DataFrame) -> bool:
    '''
    Checks the whole dataframe for any corrupt values in date_time and value columns
    '''
    try:
        if 'date_time' not in df.columns or 'value' not in df.columns:
            return False
        datetime_check = pd.to_datetime(df['date_time'], errors='coerce')
        if datetime_check.isna().any():
            return False
        value_check = pd.to_numeric(df['value'], errors='coerce')
        if value_check.isna().any():
            return False
        return True
        
    except Exception as e:
        return False
    

def cleanse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleanses a dataframe by removing rows with corrupt values in date_time and value columns.
    Returns the cleansed dataframe or None if the input dataframe is invalid.
    """
    if 'date_time' not in df.columns or 'value' not in df.columns:
        return None
    cleaned_df = df.copy()
    valid_rows = pd.Series(True, index=df.index)
    
    datetime_check = pd.to_datetime(cleaned_df['date_time'], errors='coerce')
    valid_rows &= ~datetime_check.isna()
    
    value_check = pd.to_numeric(cleaned_df['value'], errors='coerce')
    valid_rows &= ~value_check.isna()
    # deleted_rows = df[~valid_rows].copy()     #can print the deleted row
    
    cleaned_df = cleaned_df[valid_rows]
    cleaned_df['date_time'] = pd.to_datetime(cleaned_df['date_time'])
    cleaned_df['value'] = pd.to_numeric(cleaned_df['value'])
    
    if len(cleaned_df) < len(df):
        cleaned_df = cleaned_df.reset_index(drop=True)
    
    # return cleaned_df, deleted_rows   #can return the deleted row for display
    return cleaned_df
    

def validate_single_entry(date_time: Any, value: Any) -> bool:
    '''
    Checks the passed in values ( date_time and value ) for any corruption
    '''
    try:
        datetime_valid = not pd.isna(pd.to_datetime(date_time, errors='coerce'))
        value_valid = not pd.isna(pd.to_numeric(value, errors='coerce'))
        return datetime_valid and value_valid
    except Exception as e:
        return False