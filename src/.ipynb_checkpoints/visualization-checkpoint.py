# Classic python
import pandas as pd
import numpy as np
from copy import deepcopy
import os
import sys
from importlib import reload
from scipy.stats import zscore
from scipy.stats import entropy
import scipy.io
import scipy
import textwrap
from scipy import sparse
import importlib
from itertools import product
import pickle  # saving outputs
import gget

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# DMD circles
# DMD bars

def dmdPlot(dmd_res, pltArgs=None):
    """ Plot DMD eigenvalues  """
    
    if pltArgs is None:
        pltArgs = {
            'suptitle': ''
        }
    
    t = np.linspace(0, np.pi*2, 100)

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = 7, 3

    fig, axs = plt.subplots(1, 2)

    L = dmd_res['L']
    print(f"{np.real(L).max()=}")
    pdf = pd.DataFrame({'real' : np.real(L),
                        'imaginary' : np.imag(L)})

    print(f"{pdf.shape=}")
    pdf['mode'] = list(range(1, len(pdf)+1))

    # plot real parts
    sns.barplot(data=pdf, 
                x='mode',
                y='real',
                hue='mode',
                ax=axs[0],
                dodge=False,
                # ec='k',
                palette='tab20',)

    axs[0].legend().remove()

    axs[0].axhline(y=0, zorder=1, lw=0.75, c='k')
    axs[0].set_xlabel("Mode")
    axs[0].set_ylabel(r'$\mathregular{Re(\lambda)}$')


    # plot the eigenvalues
    # make the unit circle
    axs[1].plot(np.cos(t), 
             np.sin(t), 
             linewidth=1, 
             c='k',
             zorder=1)

    sns.scatterplot(data=pdf,
                    x='real', 
                    y='imaginary',
                    s=100,
                    marker=".",
                    # legend=False,
                    hue='mode',
                    ax=axs[1],
                    # ec='k',
                    palette='tab20',
                    zorder=3)

    # add the axis
    axs[1].axvline(x=0, ls=":", c='grey', zorder=0)
    axs[1].axhline(y=0, ls=":", c='grey', zorder=0)

    axs[1].set_aspect('equal')
    axs[1].set_xlabel(r'$\mathregular{Re(\lambda)}$')
    axs[1].set_ylabel(r'$\mathregular{Im(\lambda)}$')

    sns.move_legend(axs[1], 
                    title='Mode',
                    frameon=False, 
                    loc='upper right',
                    markerscale=0.5,
                    bbox_to_anchor=(1.4, 1.1))


    sns.despine()
    plt.tight_layout()
    plt.suptitle(pltArgs['suptitle'])
    plt.show()

# GO plots
def goPlot1(edf, pltArgs, k=20):
    '''
    goPlot1: plots pathways (y) by number of overlapping genes (x) with bars colored according to the adjuested p-value. Pathways are selected to have the most significant p-values.
    
    Parameters:
        edf: data frame from gget.enricher with column names 'path_name', 'name', 'numOverlap', and 'adj_p_value'. If edf is passed as a list of genes, this function will build the appropriate dataframe
        pltArgs: dictionary or string that will be the title of the plot
        k: number of pathways to consider
    
    :Code:
        db = 'GO_Biological_Process_2021'
        edf = gget.enrichr(gene_list_5, database=db)
        edf['numOverlap'] = edf['overlapping_genes'].apply(len)
        edf['name'] = edf['path_name'].apply(lambda x: x.split("(")[0].capitalize())
        goPlot1(edf, "title here", k=10)
    '''
    if type(edf) == type([]):
        gene_list = edf
        edf = gget.enrichr(gene_list, database='GO_Biological_Process_2021')
        edf['numOverlap'] = edf['overlapping_genes'].apply(len)
        edf['name'] = edf['path_name'].apply(lambda x: x.split("(")[0].capitalize())
        
    edf_sorted = edf.sort_values(by='adj_p_val')
    pdf = edf_sorted.head(k)
    print(pdf)
    pdf['name'] = pdf['path_name'].apply(lambda x: x.split("(")[0].capitalize())
    plt.figure(figsize=pltArgs['figsize'], dpi=pltArgs['dpi'])
    sns.barplot(data=pdf, 
                y='name',
                x='numOverlap',
                hue='adj_p_val',
                dodge=False,
                ec='k',
                palette='RdYlGn')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Adj_p_value')
    plt.ylabel('Pathway')
    plt.xlabel('Number of Sensor Genes')
    plt.title(pltArgs['title'])
    plt.show()
    
def goPlot2(edf, pltArgs, k=20):
    '''
    goPlot2: plots pathways (y) by number of overlapping genes (x) with bars colored according to 
             the adjusted p-value. Pathways are selected to have the maximum number of overlapping
             genes.
    
    Parameters:
        edf: data frame from gget.enricher with column names 'path_name', 'name', 'numOverlap', and
            'adj_p_value'. If edf is passed as a list of genes, this function will build the 
            appropriate dataframe
        pltArgs: dictionary or string that will be the title of the plot
        k: number of pathways to consider
    
    :Code:
        db = 'GO_Biological_Process_2021'
        edf = gget.enrichr(gene_list_5, database=db)
        edf['numOverlap'] = edf['overlapping_genes'].apply(len)
        edf['name'] = edf['path_name'].apply(lambda x: x.split("(")[0].capitalize())
        goPlot2(edf, "title here", k=10)
    '''
    if type(edf) == type([]):
        gene_list = edf
        edf = gget.enrichr(gene_list, database='GO_Biological_Process_2021')
        edf['numOverlap'] = edf['overlapping_genes'].apply(len)
        edf['name'] = edf['path_name'].apply(lambda x: x.split("(")[0].capitalize())
        
    edf_sorted = edf.sort_values(by='numOverlap', ascending=False)
    pdf = edf_sorted.head(k)
    pdf['name'] = pdf['path_name'].apply(lambda x: x.split("(")[0].capitalize())
    plt.figure(figsize=pltArgs['figsize'], dpi=pltArgs['dpi'])
    sns.barplot(data=pdf, 
                y='name',
                x='numOverlap',
                hue='adj_p_val',
                dodge=False,
                ec='k',
                palette='RdYlGn')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Adj_p_value')
    plt.ylabel('Pathway')
    plt.xlabel('Number of Sensor Genes')
    plt.title(pltArgs['title'])
    plt.show()
    
def goPlot3(left, middle, right, k=10, pltArgs=None):
    """
    goPlot3: characterizes the cell types of different segments of gene selected venn diagrams as
             stacked bar plots. Venn diagrams characterize genes expressed in different cell types
             (Fibroblast vs. HSC), identified with different thresholding criteria (mean, entropy,
             standard deviation, etc.) or between different experiments.
    
    Parameters:
        left: list of genes present in the left size of a venn diagram
        middle: list of genes present in the middle of the venn diagram
        right: list of genes present in the right of the venn diagram
        title: title of the plot
        k: number of cell types to consider
    """
    
    if pltArgs is None:
        pltArgs = {'title': None}
    
    db = 'Human_Gene_Atlas'
    dfs = {}
    dfs['L'] = gget.enrichr(left, database=db)
    dfs['L']['numOverlap'] = dfs['L']['overlapping_genes'].apply(len)
    dfs['M'] = gget.enrichr(middle, database=db)
    dfs['M']['numOverlap'] = dfs['M']['overlapping_genes'].apply(len)
    dfs['R'] = gget.enrichr(right, database=db)
    dfs['R']['numOverlap'] = dfs['R']['overlapping_genes'].apply(len)

    dfs['L'] = dfs['L'].sort_values(by='numOverlap', ascending=False)
    dfs['M'] = dfs['M'].sort_values(by='numOverlap', ascending=False)
    dfs['R'] = dfs['R'].sort_values(by='numOverlap', ascending=False)

    # reduce to the top selected cell types
    unique_pathnames = set()
    for pos in dfs.keys():
        unique_pathnames = unique_pathnames.union(set(dfs[pos]['path_name'].values[:k]))
    for pos in dfs.keys():
        dfs[pos] = dfs[pos][dfs[pos]['path_name'].isin(unique_pathnames)]

    # Build new dataframe
    df = pd.DataFrame()
    df['path_name'] = list(unique_pathnames)
    for pos in dfs.keys():
        df = pd.merge(df, dfs[pos][['path_name', 'numOverlap']], on='path_name', how='left')
    df.columns = ['Cell Type', '2015', 'Both', '2018']
    df.fillna(0, inplace=True)

    if pltArgs is None:
        pltArgs = {}
        pltArgs['size'] = (6,8)
        print('auto set size')
    
    # Make plot
    df.plot.barh(x='Cell Type', stacked=True, title=str(pltArgs['title']), figsize=pltArgs['size'])
    plt.ylabel('Cell Type')
    plt.xlabel('Number of Genes')
    plt.show()

# stability plots