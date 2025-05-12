# scr/estimation
# Hasnain method for linear time varrying and linear time invariant
import numpy as np
import pandas as pd
from scr import bioObsv

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def outputEnergy(model, trajectories):
    """
    Compute observation energy based on observed measurements.

    Params:
    --------------
    model (Model):
        An instance of LinearTimeInvariant or LinearTimeVariant.

    trajectories (np.array):
        Full state time series data of shape (genes x Time (T) x replicates).

    Returns:
    --------------
    energy (np.array):
        energy of measurements over a trajectory
    """
    if trajectories.ndim == 2:
        trajectories = trajectories[:, :, np.newaxis]

    n, T, replicates = trajectories.shape

    energy = np.zeros((replicates,))
    for t in range(T):
        measurements = model.evaluateOutput(trajectories[:, t, :])
        print(measurements.shape)
        tEnergy = (measurements * measurements).sum(axis=0)
        print(tEnergy.shape)
        print(energy.shape)
        energy += tEnergy
    return energy

def leastSquaresX0(model, trajectories, debug=False, v=False, O=None):
    """
    Predict state estimates using least squares.

    Params:
    --------------
    model (Model):
        An instance of LinearTimeInvariant or LinearTimeVariant.

    trajectories (np.array):
        Full state time series data of shape (genes x Time (T) x replicates).

    O (np.array, optional):
        precomputed observability matrix [c; ca; ca^2; ...; ca^n-1]
        
    v (bool, optional):
        command for verbose output

    debug (bool, optional):
        If True, print debugging information. Default is False.

    Returns:
    --------------
    x0estimates (np.array):
        Backward state estimates of time t=0 shape (genes x Time (T) x replicates).

    Description:
    --------------
    This function predicts the backward error, the current error, and the forward error using the least squares method.

    The input 'model' must be an instance of LinearTimeInvariant or LinearTimeVariant.
    The 'trajectories' parameter represents the full state time series data.

    The function uses least squares to estimate the states based on observations obtained from the measurement matrix C.
    """
    
    # Dimensions of the data
    n, T, numReplicates = trajectories.shape

    # extract measurements for all trajectories
    measurements_list_by_time = []
    for t in range(trajectories.shape[1]):
        output = model.evaluateOutput(trajectories[:, t, :], t=t)
        measurements_list_by_time.append(output)

    # Output variables
    x0estimates = np.zeros(trajectories.shape)

    # build observability matrix once
    if O is None:
        O = model.obsv(T+1)

    if debug:
        print(O.shape)

    # iterate over replicates
    if v:
        print('Begin Estimation')
    for rep in range(numReplicates):
        if v:
            print('replicate: ' + str(rep) + '/' + str(numReplicates))
            
        trajectory = trajectories[:, :, rep]
        # output = measurements[:, :, rep]
        outputs = np.array([])
        for t in range(T):
            if v:
                print('\t time: ' + str(t) + '/' + str(T))
            newOutput = measurements_list_by_time[t][:,rep]    # get new output
            newOutput = newOutput.reshape((newOutput.shape[0],1))
            if t != 0:
                output = np.vstack((outputs, newOutput))
            else:
                outputs = newOutput

            # compute pseudo inverse of the observability matrix
            Opinv = np.linalg.pinv(O[:outputs.shape[0], :])

            if debug:
                print(Opinv.shape)
                print(outputs.shape)
            # solve least squares problem
            x0estimates[:, t, rep] = Opinv @ outputs.squeeze()        
    return x0estimates

def leastSquares(model, trajectories, debug=False):
    """
    Predict state estimates using least squares.

    Params:
    --------------
    model (Model):
        An instance of LinearTimeInvariant or LinearTimeVariant.

    trajectories (np.array):
        Full state time series data of shape (genes x Time (T) x replicates).

    debug (bool, optional):
        If True, print debugging information. Default is False.

    Returns:
    --------------
    x0estimates (np.array):
        Backward state estimates of shape (genes x Time (T) x replicates).

    xtestimates (np.array):
        Current state estimates of shape (genes x Time (T) x replicates).

    xTestimates (np.array):
        Forward state estimates of shape (genes x Time (T) x replicates).

    Description:
    --------------
    This function predicts the backward error, the current error, and the forward error using the least squares method.

    The input 'model' must be an instance of LinearTimeInvariant or LinearTimeVariant.
    The 'trajectories' parameter represents the full state time series data.

    For LinearTimeInvariant models, it precomputes matrix powers and predicts state estimates at each time point.
    For LinearTimeVariant models, it returns a placeholder result.

    The function uses least squares to estimate the states based on observations obtained from the measurement matrix C.
    """
    # precompute matrix powers
    n, T, numReplicates = trajectories.shape
    At = {}
    Att = np.eye(n)
    if isinstance(model, bioObsv.Model.LinearTimeInvariant):
        print(type(model))
        A = model.dmd_res['A']
        for i in range(T):
            At[i] = Att
            Att = Att @ A
        C = model.output['C']
    elif isinstance(model, bioObsv.Model.LinearTimeVariant):
        print(type(model))
        C = model.LTI[0].output['C']
        At = model.phi
        result = "least squares for LinearTimeVariant model"
        return result
    else:
        raise ValueError("Unsupported model type. Supported types: LinearTimeInvariant, LinearTimeVariant")
    
    # Dimensions of the data
    n, T, numReplicates = trajectories.shape

    # extract measurements for all trajectories
    measurements_list = []
    for rep in range(trajectories.shape[2]):
        output = model.evaluateOutput(trajectories[:, :, rep])
        measurements_list.append(output)

    # Convert the list of results to a numpy array
    measurements = np.array(measurements_list)
    measurements = measurements.swapaxes(0, 1)
    measurements = measurements.swapaxes(1, 2)

    # Output variables
    x0estimates = np.zeros(trajectories.shape)
    xtestimates = np.zeros(trajectories.shape)
    xTestimates = np.zeros(trajectories.shape)

    # build observability matrix once
    O = model.obsv(T+1)

    if debug:
        print(measurements.shape)
        print(O.shape)

    # iterate over replicates
    for rep in range(numReplicates):
        trajectory = trajectories[:, :, rep]
        output = measurements[:, :, rep]

        # iterate over time
        for t in range(T):
            if debug:
                print('Observations')
            observations = output[:,0:t+1]
            if debug:
                print(f'{observations}=')
            # This transpose is very important
            observations = observations.T.ravel()
            if debug:
                print(f'{observations}=')
            Opinv = np.linalg.pinv(O[:(t+1) * C.shape[0], :])
            if debug:
                print(O[:(t+1) * C.shape[0], :].shape)
                print(Opinv.shape)
                print(observations.shape)
            x0estimates[:, t, rep] = Opinv @ observations
            xtestimates[:, t, rep] = At[t] @ x0estimates[:, t, rep]
            xTestimates[:, t, rep] = At[T - 1] @ x0estimates[:, t, rep]
            
    return x0estimates, xtestimates, xTestimates

#def obsv(A, C, T=None):
#    """
#    Generate observability matrix
#
#    Args:
#        A (np.ndarray): model state transition matrix (dynamics)
#        C (np.ndarray): model outputs (measurements)
#        T (int, optional): _description_. Defaults to None.
#
#    Returns:
#        _type_: _description_
#    """
#    p, n = C.shape
#    if T == None:
#        T = n
#    O = np.zeros((p*T, n))
#    for t in range(T):
#        O[(t-1)*p:t*p, :] = O[(t-1)*p:t*p, :] @ A
#    return O

def predictKF(model, trajectories, numItrs=10, debug=False):
    """
    Predict using Kalman Filter.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix (n x n).
    C : np.ndarray
        Measurement matrix (p x n).
    trajectories : np.ndarray
        Full state time series data (genes x Time (T) x replicates).
    numItrs : int, optional
        Number of iterations. Defaults to 10.
    debug : bool, optional
        If True, print debugging information. Defaults to False.

    Returns
    -------
    xtestimates : np.ndarray
        Array of shape (n, T, numReplicates, numItrs) representing predicted state estimates.
    """

    # Dimensions of the data
    n, T, numReplicates = trajectories.shape
    # print(trajectories.shape)

    # Output variables
    # x0estimates = np.zeros((n, T, numReplicates, numItrs))
    xtestimates = np.zeros((n, T, numReplicates, numItrs))
    # xTestimates = np.zeros((n, T, numReplicates, numItrs))

    trajectoryUnfolded = trajectories.reshape([n, T * numReplicates])
    # 1 could be a tunable parameter depending on how much we want to weight the noise
    Q = 1 * np.cov(trajectoryUnfolded)

    # construct Kalman filter
    f = KalmanFilter (dim_x=model.dmd_res['A'].shape[0], dim_z=model.output['C'].shape[0])
    f.F = model.dmd_res['A']
    f.H = model.output['C']
    f.Q = Q

    for rep in range(numReplicates):
        for itr in range(numItrs):
            # set random initial condition
            f.x = 1 * np.random.rand(model.dmd_res['A'].shape[0])
            for t in range(T):
                # get the current state estimate
                xt = f.x
                xtestimates[:, t, rep, itr] = xt
                # predict the next state
                f.predict()
                # update according to the observed outputs
                if debug:
                    print(model.output['C'].shape)
                    print(trajectories[:,t,rep].shape)
                f.update(model.output['C'] @ trajectories[:,t,rep])
        # print('Rep')
        # print(xtestimates)
            
    return xtestimates