import pandas as pd

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