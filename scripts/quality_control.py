import pandas as pd

def count_missing_values(grouped_data: dict[str, pd.DataFrame],
                          column: str
                          )->dict:
    """Summarize total rows and missing values per cell type.

    Args:
        grouped_data: Dictionary mapping cell types to DataFrames.
        column: Column in which to count missing values.

    Returns:
        Dictionary mapping each cell type to a string containing
        its total row count and the number of missing values.

    Raises:
        KeyError: If column is absent from a DataFrame.
    """

    missing_entries = {}
    for cell_type, data in grouped_data.items():
        missing_entries[cell_type] = f'total rows:{len(data)} | missing data:{len(data) - len(data.dropna(subset = column))}'
    return missing_entries

def missing_values_in_percent(grouped_data: dict[str, pd.DataFrame],
                       column: str
                       )->dict:
    """Calculate the percentage of missing values per cell type.

    Args:
        grouped_data: Dictionary mapping cell types to DataFrames.
        column: Column in which to calculate the missing-value percentage.

    Returns:
        Dictionary mapping each cell type to a descriptive string
        containing the percentage of missing values, rounded to
        two decimal places.

    Raises:
        KeyError: If column is absent from a DataFrame.
        ZeroDivisionError: If a DataFrame is empty.
    """
    
    missing_column_percent = {}
    for cell_type, data in grouped_data.items():
        missing_column_percent[cell_type] = (f'{data[column].isna().sum()/len(data)*100:.2f}% '
                                            f'of the {column} column are missing')

    return missing_column_percent

def filter_missing_values(grouped_data: dict[str,pd.DataFrame],
                     column:str,
                     )->dict[str, pd.DataFrame]:
    """Select rows with missing values in the specified column.

    Args:
        grouped_data: Dictionary mapping cell types to DataFrames.
        column: Column used to identify missing values.

    Returns:
        Dictionary mapping each cell type to a copy of the rows
        where column contains a missing value.

    Raises:
        KeyError: If column is absent from a DataFrame.
    """

    filtered_data = {}
    for cell_type, data in grouped_data.items():
        
        filtered_data[cell_type] = data[data[column].isna()].copy()

    return filtered_data

def drop_missing_values(grouped_data: dict[str, pd.DataFrame],
                      column: str = 'effect_size'
                      )->dict[str, pd.DataFrame]:
    """Remove rows with missing values in the specified column.

    Args:
        grouped_data: Dictionary mapping cell types to DataFrames.
        subset: Column used to identify missing values.
            Defaults to 'effect_size'.

    Returns:
        Dictionary mapping each cell type to a copy of its DataFrame
        containing only rows with a non-missing value in subset.

    Raises:
        KeyError: If subset is absent from a DataFrame.

    Notes:
        Input DataFrames are not modified. Missing values in other
        columns do not cause rows to be removed.
    """
    
    filtered_data = {}

    for cell_type, data in grouped_data.items():
        result = data.copy()
        filtered_data[cell_type] = result.dropna(
            subset = [column]
        )
    return filtered_data

def calculate_retained_percentage(
    grouped_data: dict[str, pd.DataFrame],
    merge_shared_data: pd.DataFrame,
) -> dict[str, str]:
    """Summarize the percentage of rows retained after merging.

    Args:
        grouped_data: Dictionary mapping cell types to the DataFrames
            used as input for the merge.
        merge_shared_data: Merged DataFrame containing shared rows,
            optionally filtered further.

    Returns:
        Dictionary mapping each cell type to a descriptive string
        containing its retained percentage, rounded to three
        decimal places.

    Raises:
        ZeroDivisionError: If an input DataFrame is empty.

    Notes:
        Assumes one row per merge key in each input DataFrame and
        that merged rows represent a subset of each input table.
        Input and merged tables must use consistent filtering.
    """
    retained_percentages = {}

    for cell_type, data in grouped_data.items():
        retained_percentages[cell_type] = (
            f'Percentage of {cell_type} rows retained after merging: '
            f'{len(merge_shared_data) / len(data) * 100:.3f}%'
        )

    return retained_percentages