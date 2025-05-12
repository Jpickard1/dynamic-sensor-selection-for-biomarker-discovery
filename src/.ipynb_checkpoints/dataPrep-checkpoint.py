import numpy as np
import pandas as pd
from anndata import AnnData
from pymatcher import matcher
import scipy as sp
from scipy.stats import entropy # entropy for Hi-C PC

def toeplitz_normalize(mat):
    n = len(mat)
    
    # Calculate mean of each off-diagonal band
    off_diagonal_means = [np.mean(np.diagonal(mat, offset=k)) for k in range(0, n)]
    
    # Construct the Toeplitz-like matrix T using vectorized operations    
    T = sp.linalg.toeplitz(off_diagonal_means)
    
    # Perform element-wise division
    result_matrix = np.divide(mat, T)
    
    return result_matrix

def pseudotimeOrdering(D):
    """
    Pseudotime Ordering

    Reorders the genes by pseudotime for a single-cell matrix.

    Parameters:
    --------------
    D (np.ndarray):
        Genes by single-cell matrix.

    Returns:
    --------------
    D_ordered (np.ndarray):
        Reordered matrix based on pseudotime.
    pseudotime (np.ndarray):
        Pseudotime values associated with the reordered matrix.
    """
    # call to pseudotime function
    m = matcher.MATCHER([D.T])
    m.infer()
    # reorder data
    pseudotime = m.master_time[0]
    sorting_order = np.argsort(pseudotime[:, 0])
    D_ordered = D[:, sorting_order]

    return D_ordered, pseudotime

def sampleTrajectories(D, timepoints=5, replicates=10):
    """
    Sample Trajectories

    Samples trajectories from the given single-cell data matrix.

    Parameters:
    --------------
    D (np.ndarray):
        Single-cell data matrix of shape (genes, samples).
    timepoints (int, optional):
        Number of timepoints to sample. Defaults to 5.
    replicates (int, optional):
        Number of replicates to generate. Defaults to 10.

    Returns:
    --------------
    dmd_data_sc (np.ndarray):
        Array of shape (genes, timepoints, replicates) representing the sampled trajectories.
    """
    n, samples = D.shape
    dmd_data_sc = np.zeros((n, timepoints, replicates))
    intervals = np.linspace(0, samples-1, timepoints)
    for rep in range(replicates):
        randomSample = [round(np.random.uniform(start, end)) for start, end in zip(intervals[:-1], intervals[1:])]
        for t, sc in enumerate(randomSample):
            dmd_data_sc[:, t, rep] = D[:, sc]
    return dmd_data_sc

def scRNAseq_quality_control(df, min_cell_counts=5900, max_cell_counts=111000, min_gene_counts=10, max_gene_counts=np.inf):
    """
    Perform quality control on a counts matrix for single cells.

    Parameters:
    - df (pd.DataFrame): Counts matrix with genes as rownames and cells as column names.
    - min_cell_counts (int): Minimum threshold for counts per cell. Cells with counts below this threshold will be removed.
    - max_cell_counts (int): Maximum threshold for counts per cell. Cells with counts above this threshold will be removed.
    - min_gene_counts (int): Minimum threshold for counts per gene. Genes with counts below this threshold will be removed.
    - max_gene_counts (int): Maximum threshold for counts per gene. Genes with counts above this threshold will be removed.

    Returns:
    - pd.DataFrame: Quality-controlled counts matrix.

    Example:
    ```python
    # Assuming 'counts_matrix' is your input counts matrix
    qc_counts_matrix = perform_quality_control(counts_matrix, min_cell_counts=500, max_cell_counts=5000, min_gene_counts=10, max_gene_counts=10000)
    ```
    """
    # Remove cells with counts below min_cell_counts or above max_cell_counts
    cell_filter = (df.sum(axis=0) >= min_cell_counts) & (df.sum(axis=0) <= max_cell_counts)
    df = df.loc[:, cell_filter]

    # Remove genes with counts below min_gene_counts or above max_gene_counts
    gene_filter = (df.sum(axis=1) >= min_gene_counts) & (df.sum(axis=1) <= max_gene_counts)
    df = df[gene_filter]

    # Additional standard filtering can be added here if needed

    return df

def filter_top_genes(df, k=100, by='entropy', remove_mitochondrial=True, force_keep_genes=None):
    """
    Filter genes to retain only the k most entropic, highly expressed, or variable genes.

    Parameters:
    - df (pd.DataFrame): Counts matrix with genes as rownames and cells as column names (after quality control).
    - k (int): Number of top genes to retain.
    - by (str): Method for selecting top genes. 'entropy' for entropy-based selection,
                'expression' for expression-based selection, 'std' for standard deviation-based selection.
    - remove_mitochondrial (bool): Whether to remove mitochondrial genes.
    - force_keep_genes (list or None): List of genes to be forced to be kept in the data.

    Returns:
    - pd.DataFrame: DataFrame with only the top k genes and forced genes.

    Example:
    ```python
    # Assuming 'qc_counts_matrix' is your quality-controlled counts matrix
    top_genes_df = filter_top_genes(qc_counts_matrix, k=100, by='entropy', remove_mitochondrial=True, force_keep_genes=['GeneA', 'GeneB'])
    ```
    """
    # Remove mitochondrial genes if specified
    if remove_mitochondrial:
        mitochondrial_genes = [gene for gene in df.index if gene.lower().startswith('mt-')]
        df = df.drop(mitochondrial_genes, axis=0, errors='ignore')

    # Select top k genes based on the specified method
    if by == 'entropy':
        gene_entropies = -(df * np.log2(df)).sum(axis=1)
        top_genes = gene_entropies.nlargest(k).index
    elif by == 'expression':
        gene_expression = df.sum(axis=1)
        top_genes = gene_expression.nlargest(k).index
    elif by == 'std':
        gene_std_rank = df.std(axis=1).rank(ascending=False, method='min')
        top_genes = gene_std_rank.nlargest(k).index
    else:
        raise ValueError("Invalid 'by' parameter. Use 'entropy', 'expression', or 'std'.")

    # Merge top k genes with forced genes
    if force_keep_genes is not None:
        genes_to_keep = set(top_genes).union(force_keep_genes)
        df = df.loc[genes_to_keep, :]
    else:
        df = df.loc[top_genes, :]

    return df

def filter_top_loci(df, k=100, by='entropy', remove_mitochondrial=True, remove_nonsense=True, ignore_columns=None):
    """
    Filter Hi-C principal components to retain only the k most entropic, central, or variable loci.

    Parameters:
    - df (pd.DataFrame): rownames are loci, column names are time points, and values are a measure of the loci at a particular time.
    - k (int): Number of top loci to retain.
    - by (str): Method for selecting top loci. 'entropy' for entropy-based selection,
                'central' for centrality-based selection, 'std' for standard deviation-based selection.
    - remove_mitochondrial (bool): Whether to remove mitochondrial genes.
    - remove_nonsense (bool): Whether to remove loci on chromosomes beyond 1,...,23 X, Y, and MT

    Returns:
    - pd.DataFrame: DataFrame with only the top k loci.
    """
    # Make a copy of the original DataFrame to avoid modifying the input
    filtered_df = df.copy()

    # Step 1: Remove mitochondrial genes if specified
    if remove_mitochondrial:
        mitochondrial_genes = ['MT']
        filtered_df = filtered_df[~filtered_df['chrom'].isin(mitochondrial_genes)]

    # Step 2: Remove loci based on the 'chrom' column if specified
    if remove_nonsense:
        valid_chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                             '21', '22', 'X', 'Y']
        filtered_df = filtered_df[filtered_df['chrom'].isin(valid_chromosomes)]

    if ignore_columns is not None:
        dfAddLater = filtered_df[ignore_columns]
        filtered_df = filtered_df.drop(ignore_columns, axis=1)
        
    # Step 3: Select top k loci based on the specified method
    if by == 'entropy':
        # Assuming 'entropy' method involves selecting loci with highest entropy across time points
        entropy_values = filtered_df.apply(lambda x: entropy(x.dropna()), axis=1)
        filtered_df = filtered_df.loc[entropy_values.nlargest(k).index]
        if ignore_columns is not None:
            dfAddLater = dfAddLater.loc[entropy_values.nlargest(k).index]
    elif by == 'central':
        # Assuming 'central' method involves selecting loci with highest centrality across time points
        # Implement centrality calculation based on your specific requirements
        # For example, you can use networkx library for graph-based centrality calculations
        total_expression = filtered_df.sum(axis=1)
        filtered_df = filtered_df.loc[total_expression.nlargest(k).index]
        if ignore_columns is not None:
            dfAddLater = dfAddLater.loc[total_expression.nlargest(k).index]
    elif by == 'std':
        # Assuming 'std' method involves selecting loci with highest standard deviation across time points
        std_values = filtered_df.std(axis=1)
        filtered_df = filtered_df.loc[std_values.nlargest(k).index]
        if ignore_columns is not None:
            dfAddLater = dfAddLater.loc[std_values.nlargest(k).index]
    else:
        raise ValueError("Invalid value for 'by'. Please choose 'entropy', 'central', or 'std'.")

    if ignore_columns is not None:
        filtered_df = pd.concat([dfAddLater, filtered_df], axis=1)
    
    # sort the rows
    filtered_df = filtered_df.sort_values(by=['chrom', 'start'])
        
    return filtered_df

def getGeneLengths(gene_table_path, gene_names):
    """A function to get gene lengths from a gene_table file

    params:
        : gene_table_path (str): path to the geneTable.csv file(pipeline output)
        : gene_names (list of str): valid gene names
    """
    gf = pd.read_csv(gene_table_path)
    gf = gf[gf['gene_name'].isin(gene_names)]
    gf = gf[gf['Feature'] == 'gene']
    gf = gf[gf['gene_biotype'] == 'protein_coding'].reset_index(drop=False)
    gf = gf[['gene_name', 'Start', 'End']].drop_duplicates()
    gf['Length'] = gf['End'] - gf['Start']
    gf = gf.groupby(['gene_name'])['Length'].max().reset_index(drop=False)
    return gf


def TPM(df, gf, target=1e6, p=1000):
    """A function to compute TPM for each column if a dataframe 
    
    params:
        : df (pd.DataFrame): the data 
        : gf (pd.DataFrame): gene lengths
        : target (float): the normalized sum of reads
        : p (int): the gene length normalization factor, default is kilo
    """
    tpm = df.copy()
    for c in tpm.columns:
        reads_per_gene  = df[c] / (gf['Length'].to_numpy() / p)
        sequence_depth = df[c].sum() / target
        tpm[c] = reads_per_gene / sequence_depth
    return tpm