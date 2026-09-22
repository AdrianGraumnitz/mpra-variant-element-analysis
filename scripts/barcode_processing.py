import pandas as pd
import mpralib.mpradata
import copy

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
) -> mpralib.mpradata.MPRABarcodeData:
    """Apply barcode-count and oligo-specific filters to MPRA data.

    The function creates a deep copy of the input data before applying
    the filters. The original object is therefore not modified.

    Args:
        element_threshold_filtered: MPRA barcode data to be filtered.
        z_score: Z-score multiplier used by the oligo-specific barcode
            filter.

    Returns:
        A filtered copy of the MPRA barcode data.
    """

    element_filtered = copy.deepcopy(element_threshold_filtered)
    element_filtered.var_filter = None

    element_filtered.apply_barcode_filter(
        mpralib.mpradata.BarcodeFilter.OLIGO_SPECIFIC,
        {"times_zscore": z_score,
        "apply_bc_threshold": False
         })
    
    return element_filtered

def dna_rna_norm(
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

