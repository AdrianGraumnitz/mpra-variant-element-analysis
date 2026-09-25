import pandas as pd
import mpralib.mpradata
import copy
import numpy as np

def threshold_filter(
        data_table: pd.DataFrame,
        threshold: int = 10
        )-> pd.DataFrame:
        """Filter oligos by their number of unique barcodes.

        Args:
            data_table: DataFrame containing the columns ``oligo_name`` and
                ``barcode``.
            threshold: Minimum number of unique barcodes required per oligo.

        Returns:
            A copy of the filtered DataFrame containing only oligos with at
            least the specified number of unique barcodes.
        """

        barcode_count = (
        data_table.groupby('oligo_name')['barcode']
        .transform('nunique')
        )

        df_filtered = data_table[barcode_count >= threshold].copy()
        return df_filtered

def barcode_filter(
    element_threshold_filtered: mpralib.mpradata.MPRABarcodeData,
    z_score: int = 3,
    threshold: int = 10
) -> mpralib.mpradata.MPRABarcodeData:
    """Apply barcode-count and oligo-specific filters to MPRA data.

    The function creates a deep copy of the input data before applying
    the filters. The original object is therefore not modified.

    Args:
        element_threshold_filtered: MPRA barcode data to be filtered.
        z_score: Z-score multiplier used by the oligo-specific barcode
            filter.
        threshold: creates a Barcode threshold attribut, doesn't remove the outliners

    Returns:
        A filtered copy of the MPRA barcode data.
    """

    element_filtered = copy.deepcopy(element_threshold_filtered)
    element_filtered.var_filter = None

    element_filtered.barcode_threshold = threshold
    element_filtered.apply_barcode_filter(
    mpralib.mpradata.BarcodeFilter.MIN_BCS_PER_OLIGO,
    {"threshold": threshold}
)

    element_filtered.apply_barcode_filter(
        mpralib.mpradata.BarcodeFilter.OLIGO_SPECIFIC,
        {"times_zscore": z_score,
        "apply_bc_threshold": True
         })
    
    return element_filtered


def get_normalized_dna_rna_counts(
        element_filtered: mpralib.mpradata.MPRABarcodeData
        )-> pd.DataFrame:
    """Create a DataFrame containing normalized DNA and RNA counts.

    Normalized DNA and RNA counts are extracted from the filtered MPRA
    barcode data. Replicate columns are numbered dynamically, allowing
    the function to handle different numbers of replicates.

    Args:
        element_filtered: Filtered MPRA barcode data containing normalized
                          DNA and RNA counts.

    Returns:
        A DataFrame containing normalized DNA and RNA counts for each
        barcode and replicate.
    """
    
    dna_norm = pd.DataFrame(data = element_filtered.normalized_dna_counts,
                            index = element_filtered.data.obs_names,
                            columns = element_filtered.data.var_names
                            )
    dna_norm = dna_norm.T
    dna_norm.columns = [f'DNA_norm_{i}'
                        for i in range(1, len(dna_norm.columns) + 1)]

    rna_norm = pd.DataFrame(data = element_filtered.normalized_rna_counts,
                            index = element_filtered.data.obs_names,
                            columns = element_filtered.data.var_names
                            )
    rna_norm = rna_norm.T
    rna_norm.columns = [f'RNA_norm_{i}'
                        for i in range(1, len(rna_norm.columns) + 1)]


    return dna_norm.join(rna_norm)

def map_oligo_to_barcode(filter_normalize_data: pd.DataFrame,
                         barcode_data: pd.DataFrame
                         )-> pd.DataFrame: 
    """Map barcodes in the DataFrame index to their corresponding oligo names.

    Adds an ``oligo_name`` column as the first column of the input DataFrame.
    Barcodes without a matching entry in ``barcode_data`` receive a missing
    value. The input DataFrame is modified in place.

    Args:
        filter_normalize_data: DataFrame containing filtered, normalized
            counts, with barcodes as its index.
        barcode_data: DataFrame containing the columns ``barcode`` and
            ``oligo_name`` used for the mapping.

    Returns:
        The modified input DataFrame with ``oligo_name`` as its first column.
    """

    filter_normalize_data['oligo_name'] = filter_normalize_data.index.map(
        barcode_data.set_index('barcode')['oligo_name'].to_dict())
    
    filter_normalize_data.insert(
    loc = 0,
    column = 'oligo_name',
    value = filter_normalize_data.pop('oligo_name') 
    )
    return filter_normalize_data

def calculate_effect_size_per_replicate(
        filter_normalize_data: pd.DataFrame
        ) :
    """Calculate the log2 RNA-to-DNA ratio for each replicate.

    Determines the number of replicates from columns containing ``DNA_norm``
    and adds an ``effect_size_i`` column for each replicate, calculated as
    log2(RNA_norm_i / DNA_norm_i). The input DataFrame is modified in place.

    Args:
        filter_normalize_data: DataFrame containing paired ``DNA_norm_i``
            and ``RNA_norm_i`` columns, numbered consecutively starting at 1.

    Returns:
        The modified input DataFrame with an additional effect-size column
        for each replicate.

    Notes:
        No pseudocount is added. Zero counts may produce infinite or
        undefined values.
    """
    
    num_replicates = filter_normalize_data.columns.str.contains('DNA_norm').sum()
    for i in range(1, num_replicates + 1): 
        filter_normalize_data[f'effect_size_{i}'] = np.log2(
            filter_normalize_data[f'RNA_norm_{i}'] 
            / filter_normalize_data[f'DNA_norm_{i}'])
    
    return filter_normalize_data