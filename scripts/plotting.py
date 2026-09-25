import matplotlib.pyplot as plt
import seaborn as sns

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
