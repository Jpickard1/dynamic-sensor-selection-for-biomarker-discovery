import pandas as pd
import numpy as np
import scipy as sp
from scr import bioObsv
from copy import deepcopy

def obsvEnergyMaximization(model, gramT=10):
    """
    Perform observer energy maximization on a LinearTimeInvariant or LinearTimeVariant model.

    Params:
    --------------
    model (Model):
        An instance of LinearTimeInvariant or LinearTimeVariant.

    gramT (int, optional):
        Number of timepoints over which to compute the Gram matrix. Default is 10.

    Returns:
    --------------
    result (dict):
        The result of observer energy maximization specific to the input model. The result includes:
        - 'sensors': DataFrame containing information about the selected sensors.
        - 'dmd': Dictionary with Dynamic Mode Decomposition (DMD) results.
        - 'G': Gram matrix.
        - 'evals': Eigenvalues of the Gram matrix.
        - 'evecs': Eigenvectors of the Gram matrix.

    References:
    --------------
    Hasnain, A., Balakrishnan, S., Joshy, D. M., Smith, J., Haase, S. B., & Yeung, E. (2023).
    Learning perturbation-inducible cell states from observability analysis of transcriptome dynamics.
    Nature Communications, 14(1), 3148. [Nature Publishing Group UK London]
    """
    if isinstance(model, bioObsv.Model.LinearTimeInvariant):
        # Perform observer energy maximization for LinearTimeInvariant model
        # You can access model-specific attributes and methods here
        # A = model.dmd_res['Atilde']
        # u = model.dmd_res['u_r']
        # x0_embedded = model.dmd_res['data_embedded'][:,0,:]
        G = model.gram_matrix(T=gramT, reduced=True)
    elif isinstance(model, bioObsv.Model.LinearTimeVariant):
        # Perform observer energy maximization for LinearTimeVariant model
        # You can access model-specific attributes and methods here
        # need to implement for time invariant models
        G = model.gram_matrix(T=gramT)
    else:
        raise ValueError("Unsupported model type. Supported types: LinearTimeInvariant, LinearTimeVariant")
        
    D, V = np.linalg.eig(G)
    V = np.abs(V) # this line was missing. it is used in the original Hasnain code.

    # this line we will change based on the annotated data object
    obs = pd.DataFrame({'state'   : model.states,
                        'ev1'    : V[:,0],
                        'weight' : np.real(V[:,0])})

    obs['rank'] = obs['weight'].rank(ascending=False)
    obs = obs.sort_values(by='rank', ascending=True)
    obs = obs.reset_index(drop=True)

    return {'sensors': obs,
            'G'      : G,
            'evals'  : D,
            'evecs'  : V}

def energyMaximizationTV(model, times, v=False):
    if not isinstance(model, bioObsv.Model.LinearTimeInvariant) and not isinstance(model, bioObsv.Model.LinearTimeVariant):
        raise ValueError("Unsupported model type. Supported types: LinearTimeInvariant, LinearTimeVariant")
    TVSensors = {}
    for t in times:
        if v:
            print('t: ' + str(t) + '/' + str(len(times)))
        G = model.gram_matrix_TV(T=t)
        # print(G.shape)
        D, V = sp.sparse.linalg.eigs(G, k=1)
        # D, V = np.linalg.eig(G)
        V = np.abs(V) # this line was missing. it is used in the original Hasnain code.
        obs = pd.DataFrame({'state'   : model.states,
                            'ev1'    : V[:,0],
                            'weight' : np.real(V[:,0])})
        obs['rank'] = obs['weight'].rank(ascending=False)
        obs = obs.sort_values(by='rank', ascending=True)
        obs = obs.reset_index(drop=True)
        TVSensors[t] = {'sensors': obs,
                        'G'      : G,
                        'evals'  : D,
                        'evecs'  : V}
    return TVSensors

def submodularSensorSelection(A, gramT=1,maxSensors=2, subCriteria=1):
    n = A.shape[0]
    # Submodular optimization
    S = []              # selected sensors
    R = list(range(n))  # remaining sensors

    # while selecting more sensors
    while len(S) < maxSensors:
        M = np.zeros(len(R))  # save scores for each sensor
        # try each of the remaining sensors
        for i, vx in enumerate(R):
            C = getC(n, np.append(S, vx))  # create C matrix
            G = np.zeros_like(A)           # construct new gramian
            for t in range(gramT):         # vary finite time
                G += np.dot(np.dot(A.T, C.T), np.dot(C, A))
            if subCriteria == 1:
                M[i] = np.trace(G)      # Four measures of submodularity
            elif subCriteria == 2:
                M[i] = np.trace(np.linalg.inv(G))
            elif subCriteria == 3:
                M[i] = np.log(np.linalg.det(G))
            elif subCriteria == 4:
                M[i] = np.linalg.matrix_rank(G)
        vx = np.argmax(M)  # find highest weighted next sensor
        S.append(R[vx])    # select the next sensor
        R.pop(vx)          # remove sensor from remaining vertices
    return S

def getC(n, idxs):
    # Define your getC function if it's not already defined
    C = sp.sparse(n, len(idxs))
    for i in range(idxs):
        C[i, idxs[i]] = 1
    return C


