import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
import pickle


tmap = {
    'S1a' : 0,
    'S1b' : 0,
    'S2a' : 0,
    'S2b' : 0,
    'S3a' : 1,
    'S3b' : 1,
    'S4a' : 2,
    'S4b' : 2,
    'S5a' : 3,
    'S5b' : 3,
    'S6a' : 4,
    'S6b' : 4,
    'S7a' : 5,
    'S7b' : 5,
    'S8a' : 6,
    'S8b' : 6,
    'S9a' : 7,
    'S9b' : 7,
}

rmap = {
    'S1a' : "r1c",
    'S1b' : "r2c",
    'S2a' : "r1",
    'S2b' : "r2",
    'S3a' : "r1",
    'S3b' : "r2",
    'S4a' : "r1",
    'S4b' : "r2",
    'S5a' : "r1",
    'S5b' : "r2",
    'S6a' : "r1",
    'S6b' : "r2",
    'S7a' : "r1",
    'S7b' : "r2",
    'S8a' : "r1",
    'S8b' : "r2",
    'S9a' : "r1",
    'S9b' : "r2",
}

cmap = {
    'S1a' : "control",
    'S1b' : "control",
    'S2a' : "timecourse",
    'S2b' : "timecourse",
    'S3a' : "timecourse",
    'S3b' : "timecourse",
    'S4a' : "timecourse",
    'S4b' : "timecourse",
    'S5a' : "timecourse",
    'S5b' : "timecourse",
    'S6a' : "timecourse",
    'S6b' : "timecourse",
    'S7a' : "timecourse",
    'S7b' : "timecourse",
    'S8a' : "timecourse",
    'S8b' : "timecourse",
    'S9a' : "timecourse",
    'S9b' : "timecourse",
}


s_genes = [
    'MCM5','PCNA','TYMS','FEN1','MCM7','MCM4',
    'RRM1','UNG','GINS2','MCM6','CDCA7','DTL','PRIM1',
    'UHRF1','CENPU','HELLS','RFC2','POLR1B','NASP',
    'RAD51AP1','GMNN','WDR76','SLBP','CCNE2','UBR7',
    'POLD3','MSH2','ATAD2','RAD51','RRM2','CDC45',
    'CDC6','EXO1','TIPIN','DSCC1','BLM','CASP8AP2',
    'USP1','CLSPN','POLA1','CHAF1B','MRPL36','E2F8'
]

g2_genes = [
    'HMGB2','CDK1','NUSAP1','UBE2C','BIRC5',
    'TPX2','TOP2A','NDC80','CKS2','NUF2',
    'CKS1B','MKI67','TMPO','CENPF','TACC3',
    'PIMREG','SMC4','CCNB2','CKAP2L','CKAP2',
    'AURKB','BUB1','KIF11','ANP32E','TUBB4B',
    'GTSE1','KIF20B','HJURP','CDCA3','JPT1',
    'CDC20','TTK','CDC25C','KIF2C','RANGAP1',
    'NCAPD2','DLGAP5','CDCA2','CDCA8','ECT2',
    'KIF23','HMMR','AURKA','PSRC1','ANLN',
    'LBR','CKAP5','CENPE','CTCF','NEK2',
    'G2E3','GAS2L3','CBX5','CENPA'
]

def leastSquaresX0new(model, trajectories, debug=False, v=False, O=None, times=None, reduced=False):
    # Dimensions of the data
    n, T, numReplicates = trajectories.shape
    if times is None:
        times = np.arange(T)
    if debug:
        print(f'trajectories.shape={trajectories.shape}')
    # extract measurements for all trajectories
    measurements_list_by_time = []
    for t in range(trajectories.shape[1]):
        # print(t)
        output = model.evaluateOutput(trajectories[:, t, :], t=t)
        measurements_list_by_time.append(output)
    if debug:
        print('Output shapes:')
        for t in range(T):
            print('\tt=' + str(t) + ', ' + str(measurements_list_by_time[t].shape))

    # Output variables
    x0estimates = np.zeros(trajectories.shape)

    # build observability matrix once
    if O is None:
        O = model.obsv(T+1, reduced=reduced)

    if debug:
        print('Observability Matrix shape: ' + str(O.shape))

    # iterate over replicates
    if v:
        print('Begin Estimation')
    for rep in range(numReplicates):
        if v:
            print('replicate: ' + str(rep) + '/' + str(numReplicates))
        trajectory = trajectories[:, :, rep]
        
        for t in range(T):
            if v:
                print('\t time: ' + str(t) + '/' + str(T))
            newOutput = measurements_list_by_time[t][:,rep]    # get new output
            newOutput = newOutput.reshape((newOutput.shape[0],1))
            if t != 0:
                outputs = np.vstack((outputs, newOutput))
            else:
                outputs = newOutput
            if debug:
                print('outputs.shape='+str(outputs.shape))
            # compute pseudo inverse of the observability matrix
            if t in times:
                Opinv = np.linalg.pinv(O[:outputs.shape[0], :])
                if debug:
                    print('Opinv.shape='+str(Opinv.shape))
                    print('outputs.squeeze().shape='+str(outputs.squeeze().shape))
                if outputs.shape[0] != Opinv.shape[1]:
                    x0estimates[:, t, rep] = Opinv @ outputs.squeeze()
                else:
                    x0estimates[:, t, rep] = (Opinv @ outputs).squeeze()
    return x0estimates

def meltDf(df):
    """A custom function to transform the raw data
    representation into a melted one """
    df = pd.melt(df, id_vars='geneName')
    df['timepoint'] = df['variable'].map(tmap)
    df['replicate'] = df['variable'].map(rmap)
    df['control'] = df['variable'].map(cmap)
    df['hours'] = df['timepoint'] * 8
    return df


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

def getGenePos(gf, gene_names):
    """A function to get gene lengths from a gene_table file

    params:
        : gene_table_path (str): path to the geneTable.csv file(pipeline output)
        : gene_names (list of str): valid gene names
    """
    gf = gf[gf['gene_name'].isin(gene_names)]
    gf = gf[gf['Feature'] == 'gene']
    gf = gf[gf['gene_biotype'] == 'protein_coding'].reset_index(drop=False)
    gf = gf[['gene_name', 'Chromosome', 'Start', 'End']].drop_duplicates()
    gf['Length'] = gf['End'] - gf['Start']
    chromLengths = {}
    for x in gf.groupby('Chromosome'):
        chromLengths[x[0]] = max(x[1]['End'].values)
    absLocation = {}
    absLoc = 0
    for chrom, lens in chromLengths.items():
        absLocation[chrom] = absLoc
        absLoc += lens
    gf['Abs Location'] = gf['Start']
    for index, row in gf.iterrows():
        gf.loc[index, 'Abs Location'] += absLocation[gf.loc[index, 'Chromosome']]
    return gf

def sortRNAgeneByLoc(cdf, gf):
    locs = []
    genes = []
    for gene, _ in cdf.iterrows():
        locs.append(gf.loc[gene, 'Abs Location'])
        genes.append(gene)
    idxs = np.argsort(locs)
    sorted_cdf = cdf.iloc[idxs]
    return sorted_cdf

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


def CPM(df, target=1e6):
    """A function to compute CPM for each column if a dataframe 
    
    params:
        : df (pd.DataFrame): the data 
        : target (float): the normalized sum of reads
    """
    cpm = df.copy()
    for c in cpm.columns:
        cpm[c] = (df[c] / df[c].sum()) *  target
    return cpm


def getFCFrame(df, cols):
    """A function to return a new dataframe with 
    foldchanges over the initial. Expects cols to be time-ordered
    and that cols[0] is the initial condition """
    df2 = df[cols].copy()
    x0 = cols[0] # initial condition
    for c in cols[1:]:
        df2[c] = (df2[c] + 1) /  (df2[x0].mean() + 1)
    return df2[cols[1:]].copy()

def getFCFrame2(df, cols):
    """A function to return a new dataframe with 
    foldchanges over the initial. Expects cols to be time-ordered
    and that cols[0] is the initial condition """
    df2 = df[cols].copy()
    x0 = cols[0] # initial condition
    for c in cols[1:]:
        df2[c] = (df2[c] + 1) /  (df2[x0] + 1)
    return df2[cols[1:]].copy()

def getDMDdata(data_type, filepath):
    # this pickle file was created in the notebook reloadingData.ipynb
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data.get(data_type)

def data2DMD(df, rescale=True):
    """A function to make DMD suitable data from the 2015 data"""
    a = [x for x in df.columns if "a" in x]
    b = [x for x in df.columns if "b" in x]

    if rescale:
        dfa = getFCFrame2(df, a)
        dfb = getFCFrame2(df, b)

    else:
        dfa = df[a[1:]]
        dfb = df[b[1:]]
    
    dmd_data = np.asarray([dfa, dfb])
    dmd_data = np.swapaxes(dmd_data, 0, 2)
    dmd_data = np.swapaxes(dmd_data, 0, 1)

    return dmd_data


def data2DMD2017(df, rescale=True):
    """A function to make DMD suitable data from the 2017 data"""
    a = [x for x in df.columns if "R1" in x]
    b = [x for x in df.columns if "R2" in x]

    if rescale:
        dfa = getFCFrame2(df, a)
        dfb = getFCFrame2(df, b)
    else:
        dfa = df[a[1:]]
        dfb = df[b[1:]]

    dmd_data = np.asarray([dfa, dfb])
    dmd_data = np.swapaxes(dmd_data, 0, 2)
    dmd_data = np.swapaxes(dmd_data, 0, 1)

    return dmd_data


def getMuData(df):
    """A function to get the mean expression of each replicate at
    each time point """
    mu_data = df.copy()
    mu_data = mu_data.T

    mu_data = mu_data.reset_index(drop=False)
    mu_data['time'] = mu_data['index'].apply(lambda x: x.replace("a", "").replace("b", "").replace('S', ''))
    mu_data['time'] = mu_data['time'].astype(int)
    mu_data = mu_data.drop(columns=['index'])
    mu_data = mu_data.groupby('time').mean().reset_index(drop=False)
    mu_data = mu_data.sort_values(by='time')
    mu_data = mu_data.set_index('time')
    return mu_data

