import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_barcode_per_oligo(
        barcode_per_oligo_tabels: list,
        cell_names: list
        ):
    """Plot barcode-count distributions as one violin plot per cell type.

    Args:
        barcode_per_oligo_tabels: List of at least two DataFrames, each
            containing one row per oligo and a numeric ``n_barcodes`` column.
        cell_names: Cell-type labels in the same order as the tables.
            Must contain one label per table.

    Returns:
        None. Creates a Matplotlib figure with side-by-side violin plots.

    Notes:
        The input tables are not modified. The figure is neither explicitly
        displayed nor saved; display depends on the plotting environment.
    """
    
    fig, axes = plt.subplots(1,
                             len(barcode_per_oligo_tabels), 
                                figsize = (5 * len(barcode_per_oligo_tabels), 5))
    for table, ax, cell_name in zip(barcode_per_oligo_tabels, axes, cell_names):
        sns.violinplot(
            data=table,
            y='n_barcodes',
            ax=ax
        )
        ax.set_ylabel('Barcodes per oligo')
        ax.set_title(cell_name)

def cell_variance_box_plot(
        cell_tables: list[pd.DataFrame],
        filter_names: list[str] = []
        ):
    """Plot cell-type-specific variance distributions for multiple tables.

    Args:
        cell_tables: DataFrames containing ``cell_type`` and ``variance``
            columns. Each table is plotted in a separate subplot.
        filter_names: Titles for the subplots, in the same order as
            ``cell_tables``. Provide one title per table.

    Returns:
        None.

    Notes:
        Boxplots use notches and hide outlier markers. Hidden outliers
        remain part of the data used to calculate boxplot statistics.
        Input DataFrames are not modified.
    """
    
    fig, axes = plt.subplots(1,
                             len(cell_tables),
                             figsize = (5 * len(cell_tables), 5))

    axes = np.atleast_1d(axes)
    for table, ax, filter_name in zip(cell_tables, axes, filter_names):
        sns.boxplot(data = table,
                    x = 'cell_type',
                    y = 'variance',
                    ax = ax,
                    showfliers=False,
                    notch=True)
        ax.set_ylabel('variance')
        ax.set_title(table['cell_type'])
        ax.set_title(filter_name)
