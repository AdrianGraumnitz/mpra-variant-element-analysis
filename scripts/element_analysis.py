import pandas as pd
import re

def effect_sizes_long(
        barcode_oligo_effect_sizes: pd.DataFrame
        )-> pd.DataFrame:
    """Reshape replicate effect sizes from wide to long format.

    Args:
        barcode_oligo_effect_sizes: DataFrame containing ``oligo_name`` and
            ``effect_size_i`` columns, numbered consecutively starting at 1.

    Returns:
        A DataFrame with columns ``oligo_name``, ``replicate``, and
        ``effect_size``, containing one row per input row and replicate.

    Notes:
        The original index is discarded. The input DataFrame is not modified.
    """

    num_rep = barcode_oligo_effect_sizes.columns.str.contains('effect_size_').sum()
    num_rep_array = [f'effect_size_{i}' for i in range(1,num_rep+1)]

    return barcode_oligo_effect_sizes.melt(
        id_vars = 'oligo_name',
        value_vars = num_rep_array,
        var_name='replicate',
        value_name = 'effect_size'
        
    )

def aggregate_effect_size_mean_var_per_oligo(reshape_replicate_data: pd.DataFrame
                                             )-> pd.DataFrame:
    """Calculate the mean and variance of effect sizes per oligo.

    Args:
        reshape_replicate_data: Long-format DataFrame containing
            ``oligo_name`` and ``effect_size`` columns.

    Returns:
        A DataFrame with one row per oligo and columns ``oligo_name``,
        ``mean``, and ``var``.

    Notes:
        All effect sizes belonging to the same oligo are pooled.
        Missing effect sizes are excluded from the calculations.
        Variance is calculated as sample variance (ddof=1) and is
        undefined for groups with fewer than two valid values.
        The input DataFrame is not modified.
    """
    
    return (
        reshape_replicate_data.groupby('oligo_name')['effect_size'].agg(
            ['mean', 'var'])
        ).reset_index()

def rename_aggregate_columns(
        aggregate_tabel: pd.DataFrame,
        cell_type: str
    )-> pd.DataFrame:
    """Append the cell type to the mean and variance column names.

    Args:
        aggregate_table: DataFrame containing ``mean`` and ``var`` columns.
        cell_type: Cell-type label to append to the column names.

    Returns:
        A new DataFrame with ``mean`` renamed to ``mean_{cell_type}``
        and ``var`` renamed to ``var_{cell_type}``.
        All other columns retain their names.
    """

    return aggregate_tabel.rename(columns = {
        'mean': f'mean_{cell_type}',
        'var': f'var_{cell_type}'
    })

def merge_aggregate_tables(
        aggregate_tables: list[pd.DataFrame],
        how: str = 'outer',
        on: str | list[str] = 'oligo_name'
        ) -> pd.DataFrame:

    """Merge aggregated tables using outer joins on a shared key.

    Args:
        aggregate_tables: Non-empty list of DataFrames to merge, typically
            containing statistics with cell-type-specific column names.
        on: Shared column used as the merge key. Defaults to ``oligo_name``.

    Returns:
        A DataFrame containing all merge keys from all input tables.
        Missing matches are filled with NaN.

    Notes:
        Merge keys should be unique within each table to avoid multiplying
        rows when matching duplicate keys. Input DataFrames are not modified.

    Raises:
        IndexError: If ``aggregate_tables`` is empty.
    """
    merged = aggregate_tables[0].copy()

    for table in aggregate_tables[1:]:
        merged = merged.merge(
            table,
            how = how,
            on = on
        )

    return merged

def reshape_celltype_variance(
        var_table: pd.DataFrame,
        cell_types: list
        )-> pd.DataFrame:
    """Reshape cell-type-specific effect-size variances to long format.

    Args:
        var_table: DataFrame containing ``oligo_name`` and a
            ``var_{cell_type}`` column for each requested cell type.
        cell_types: Cell-type labels to include, such as
            ``["NGN2", "WTC11"]``.

    Returns:
        A DataFrame with columns ``oligo_name``, ``cell_type``, and
        ``variance``, containing one row per input row and requested
        cell type. Cell-type labels have the ``var_`` prefix removed.

    Notes:
        Existing variance values are reshaped without recalculation.
        Missing values are retained. The original index and unselected
        columns are discarded. The input DataFrame is not modified.

    Raises:
        KeyError: If ``oligo_name`` or a requested variance column
            is missing.
    """
    
    var_long_table = var_table.melt(
        id_vars=['oligo_name'],
        value_vars= [f'var_{i}' for i in cell_types],
        var_name = 'cell_type',
        value_name = 'variance')
    var_long_table['cell_type'] = (
        var_long_table['cell_type']
        .str.removeprefix('var_')
    )
    return var_long_table

def reshape_celltype_mean(
        mean_table: pd.DataFrame,
        cell_types: list
        )-> pd.DataFrame:
    """Reshape cell-type-specific mean effect sizes to long format.

    Args:
        mean_table: DataFrame containing ``oligo_name`` and a
            ``mean_{cell_type}`` column for each requested cell type.
        cell_types: Cell-type labels to include, such as
            ``["NGN2", "WTC11"]``.

    Returns:
        A DataFrame with columns ``oligo_name``, ``cell_type``, and
        ``mean``, containing one row per input row and requested
        cell type. Cell-type labels have the ``mean_`` prefix removed.

    Notes:
        Existing mean values are reshaped without recalculation.
        Missing values are retained. The original index and unselected
        columns are discarded. The input DataFrame is not modified.

    Raises:
        KeyError: If ``oligo_name`` or a requested mean column
            is missing.
    """
    
    mean_long_table = mean_table.melt(
        id_vars=['oligo_name'],
        value_vars= [f'mean_{i}' for i in cell_types],
        var_name = 'cell_type',
        value_name = 'mean')
    mean_long_table['cell_type'] = (
        mean_long_table['cell_type']
        .str.removeprefix('mean_')
    )
    return mean_long_table

def group_to_oligo(snakeflow_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate barcode-level counts per oligo.

    Args:
        snakeflow_data: Barcode-level DataFrame containing ``oligo_name``,
            ``barcode``, and count columns named ``dna_count_<number>``
            or ``rna_count_<number>``. Replicate numbers do not need
            to be consecutive.

    Returns:
        A DataFrame with one row per oligo, summed counts for each
        detected count column, and ``unique_barcode_count`` containing
        the number of distinct non-missing barcodes per oligo.

    Raises:
        ValueError: If no matching DNA or RNA count columns are found.
        KeyError: If ``oligo_name`` or ``barcode`` is missing.

    Notes:
        Rows with missing ``oligo_name`` are excluded.
        Missing count values are skipped when summing; groups with
        only missing values receive a sum of zero.
        The input DataFrame is not modified.
    """
    count_columns = [
        column
        for column in snakeflow_data.columns
        if re.fullmatch(r"(dna|rna)_count_\d+", column)
    ]

    if not count_columns:
        raise ValueError("Keine DNA- oder RNA-Count-Spalten gefunden.")

    # dictionary for the agg function
    aggregations = {
        column: (column, "sum")
        for column in count_columns
    }
    aggregations["unique_barcode_count"] = ("barcode", "nunique")

    return (
        snakeflow_data
        .groupby("oligo_name", as_index=False)
        .agg(**aggregations) # the two stars unpacking the aggregation dictionary to arguments for the agg function
    )

def group_cells_to_oligo(
    input_snakeflow_files: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Load and aggregate barcode-level datasets per cell type.

    Args:
        input_snakeflow_files: Mapping of cell-type names to paths
            of tab-separated barcode-level count files.

    Returns:
        A dictionary mapping each cell type to its aggregated
        oligo-level DataFrame.
    """
    grouped_data = {}

    for cell_type, file_path in input_snakeflow_files.items():
        barcode_data = pd.read_table(file_path)
        grouped_data[cell_type] = group_to_oligo(
            barcode_data
        )

    return grouped_data

def mapping(
    grouped_data: dict[str, pd.DataFrame],
    mapping_data: pd.DataFrame,
    on: str,
    element: str,
    output_column:str,
    target_on: str = "oligo_name"
    ) -> dict[str, pd.DataFrame]:
    """Map a value column from a shared lookup table to each dataset.

    Args:
        grouped_data: Dictionary mapping cell-type names to DataFrames
            containing the column specified by ``target_on``.
        mapping_data: Shared lookup table containing the columns
            specified by ``on`` and ``element``. Values in ``on``
            must be unique.
        on: Key column in the lookup table.
        element: Source column containing the values to map.
        output_column: Name of the column to add or overwrite
            in each target DataFrame.
        target_on: Key column in the target DataFrames.
            Defaults to ``oligo_name``.

    Returns:
        The input dictionary with ``output_column`` added or
        overwritten in each DataFrame. Keys without a matching
        lookup entry receive a missing value.

    Raises:
        KeyError: If a required column is missing.
        pandas.errors.InvalidIndexError: If lookup keys are not unique.

    Notes:
        Uses the same lookup table for every cell type.
        Modifies the DataFrames in the input dictionary in place.
    """
    mapping_element = mapping_data.set_index(on)[element]

    for cell_type, _ in grouped_data.items():
        grouped_data[cell_type][output_column] = (
            grouped_data[cell_type][target_on].map(mapping_element)
        )

    return grouped_data

def map_column_from_files(
    grouped_data: dict[str, pd.DataFrame],
    files: dict[str, str],
    on: str,
    element: str,
    output_column: str,
    target_on: str = "oligo_name",
) -> dict[str, pd.DataFrame]:
    """Map a column from cell-type-specific TSV files to each dataset.

    Args:
        grouped_data: Dictionary mapping cell-type names to DataFrames
            containing the column specified by ``target_on``.
        files: Dictionary mapping cell-type names to TSV file paths.
        on: Column in each TSV file containing unique lookup keys.
        element: Column in each TSV file containing the values to map.
        output_column: Name of the column to add or overwrite.
        target_on: Column in each target DataFrame containing the keys
            to match against ``on``.

    Returns:
    A new dictionary containing copies of the input DataFrames with
    ``output_column`` added or overwritten. Unmatched keys receive
    missing values.


    Raises:
        KeyError: If a cell type has no corresponding file path or a
            required column is missing.
        pandas.errors.InvalidIndexError: If lookup keys are not unique.
        FileNotFoundError: If a specified file does not exist.

    Notes:
    The input dictionary and its DataFrames are not modified.
    All existing rows are retained.
    """
    mapped = {}
    for cell_type, data in grouped_data.items():
        result = data.copy()
        mapped_data = mapping(
            grouped_data={cell_type: result},
            mapping_data=pd.read_table(files[cell_type]),
            on=on,
            element=element,
            output_column=output_column,
            target_on=target_on,
        )
        result = mapped_data[cell_type]
        mapped[cell_type] = result

    return mapped

def mean_for_label(
    grouped_data: pd.DataFrame,
    label: str = "scramble_ctrl",
    column: str = "effect_size",
) -> float:
    """Calculate the mean of a numeric column for a specific label.

    Args:
        data: DataFrame containing a ``label`` column and the numeric
            column to summarize.
        label: Label identifying the rows to include.
        column: Name of the numeric column to summarize.

    Returns:
        The mean of the selected values, or NaN if no valid values
        are available.

    Notes:
        Missing values in the selected column are excluded.
        The input DataFrame is not modified.

    Raises:
        KeyError: If ``label`` or the requested column is missing.
    """
    return grouped_data.loc[grouped_data["label"] == label, column].mean()


def sd_for_label(
    grouped_data: pd.DataFrame,
    label: str = "scramble_ctrl",
    column: str = "effect_size",
) -> float:
    """Calculate the sample standard deviation for a specific label.

    Args:
        data: DataFrame containing a ``label`` column and the numeric
            column to summarize.
        label: Label identifying the rows to include.
        column: Name of the numeric column to summarize.

    Returns:
        The sample standard deviation of the selected values, or NaN
        if fewer than two valid values are available.

    Notes:
        Missing values in the selected column are excluded.
        Uses the sample standard deviation with ddof=1.
        The input DataFrame is not modified.

    Raises:
        KeyError: If ``label`` or the requested column is missing.
    """
    return grouped_data.loc[grouped_data["label"] == label, column].std()

def calculate_normalized_values(grouped_data: dict[str, pd.DataFrame],
                                     label: str = 'scramble_ctrl',
                                     column: str = 'effect_size'
                                     )-> dict[str, pd.DataFrame]:
    """Normalize values using control statistics separately per cell type.

    Args:
        grouped_data: Dictionary mapping cell-type names to DataFrames
            containing a ``label`` column and the requested numeric column.
        label: Label identifying the control rows.
        column: Name of the numeric column to normalize.

    Returns:
        A new dictionary containing copies of the input DataFrames with
        ``norm_effect_size`` added or overwritten.

    Notes:
        Normalization is calculated as (value - control_mean) / control_sd.
        Control statistics are calculated separately for each DataFrame.
        Missing control values are excluded from these calculations.
        The control standard deviation uses ddof=1.
        A zero or undefined control standard deviation can produce
        infinite or missing normalized values.
        The input dictionary and its DataFrames are not modified.

    Raises:
        KeyError: If ``label`` or the requested column is missing.
    """
    
    normalized_data = {}
    
    for cell_type, data in grouped_data.items():
        result = data.copy()
        control_mean = mean_for_label(grouped_data = data,
                                      label = label,
                                      column = column)
        control_sd = sd_for_label(grouped_data = data,
                                  label = label,
                                  column = column)
        result['norm_effect_size'] = (data[column] - control_mean)/control_sd
        normalized_data[cell_type] = result

    return normalized_data