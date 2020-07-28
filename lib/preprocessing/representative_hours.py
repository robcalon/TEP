import numpy as np
import pandas as pd
import pulp

def validation(series, sslice=24, nmslice=[0,365]):
    """
    Function to check the passed time series.
    
    Parameters
    ----------
    series : list
        List of time series to include. The series should be of dtype ndarray.
    sslice : int, default 24
        Size of a slice. Specifies the size of the slice of a representative day.
    nmslice : list, default [0, 365]
        Slice range to include in the evaluation. Can be used to work with 
        series of differentlengths.

    Returns
    -------
    cseries : list
        Returns a list with converted time series. See conversion function. 
    """        

    # check if input is a list
    if isinstance(series, list):
        
        # check if all elements in list are ndarrays
        if all(isinstance(item, np.ndarray) for item in series):
            
            return converter(series, sslice=sslice, nmslice=nmslice)
        
        else:
            # locate error position
            position = [isinstance(item, np.ndarray) for item in series].index(False)   

            # return error message
            raise KeyError('Series of type ' + str([type(series[position])]) + ' are unsupported')
    
    else:
        # raise error if anything else than list is passed
        raise KeyError('Series must be passed in list-form')


def slicer(array, sslice=24, nmslice=[0, 365]):
    """
    Function to take a slice from an array.
    
    Parameters
    ----------
    array : ndarray
        1D array that is sliced.
    sslice : int, default 24
        Size of a slice. Specifies the size of the slice of a representative day.
    nmslice : list, default [0, 365]
        Slice range to include in the evaluation. Can be used to work with 
        series of differentlengths.

    Returns
    -------
    sarray : ndarray
        Returns a sliced ndarray
    """    
    
    start, end = sslice * np.array(nmslice)
    
    # ravel 1D arrays
    array = array.ravel()
    
    # check for nan values
    if np.isnan(np.sum(array)) == 0:

        # check if size matches
        if array.size >= end:        

            # append array to subsets
            return array[start:end]

        else:
            # when array is too short 
            raise ValueError('Size of Array is smaller than the number of slices')

    else:
        # when array contains NaN values
        raise ValueError('Array containts NaN values')


def converter(series, sslice=24, nmslice=[0, 365]):
    """
    Function to convert a list of time series into a list of
    equally sized arrays.
    
    Parameters
    ----------
    series : list
        List of time series to convert. 
    sslice : int, default 24
        Size of a slice. Specifies the size of the slice of a representative day.
    nmslice : list, default [0, 365]
        Slice range to include in the evaluation. Can be used to work with 
        series of different lengths.

    Returns
    -------
    cseries : list
        Returns a list with converted time series.
    """
    
    cseries = []
    
    for array in series:

        if len(array.shape) == 1:
            
            sarray = slicer(array, sslice=sslice, nmslice=nmslice)
            cseries.append(sarray)
        
        elif len(array.shape) > 2:
            raise ValueError('Can only process 1D and 2D numpy arrays')
        
        elif array.shape[1] == 1:
            sarray = slicer(array, sslice=sslice, nmslice=nmslice)
            cseries.append(sarray)
        
        else:
            subseries = np.hsplit(array, array.shape[1])
            cseries += converter(subseries, sslice=sslice, nmslice=nmslice)

    return cseries


def matrices(series, rdays, nbins=5, sslice=24):
    """
    Function that calculates the L and A matrices of a given series for 
    a given set of reference days.
    
    Parameters
    ----------
    series : ndarray
        Series for which the matrices are calculated.
    rdays : ndarray
        1D array specifying the days that are evaluated in the optimization.
    nbins : int, default 5
        Number of bins to use to split the series data.
    sslice : int, default 24
        Size of a slice. Specifies the size of the slice of a representative day.
        
    Returns
    -------
    A : ndarray
        2D array with A-matrix values
    L : ndarray
        1D array with L-matrix values
    """
    
    # Specify empty array for later use
    A = np.zeros(shape=(rdays.size, nbins))
    
    # Set L matrix by binning the time series
    L = np.histogram(series, bins=nbins)[0] / series.size
    
    # Slice the random days from the time series
    for index, rday in enumerate(rdays):
        
        # start and end of slice
        start = rday * sslice
        end = start + sslice

        # Bin the results of the selected random day
        row = np.histogram(series[start:end], bins=nbins, range=(series.min(),series.max()))[0]
         
        # Set the nth row of the A matrix
        A[index] = np.divide(row, sslice)
        
    return A, L


def solver(series, rdays, nbins=5, sslice=24, nslice=365):
    """
    Function that optimizes the relative error of a given set of random days.
    
    Parameters
    ----------
    series : list
        List of series containing int or float type values.     
    rdays : ndarray
        1D array specifying the days that are evaluated in the optimization.
    nbins : int, default 5
        Number of bins to use to split the series data.
    sslice : int, default 24
        Size of a slice. Specifies the size of the slice of a representative day.
    nslice : int, default 365
        Number of considered slices.
        
    Returns
    -------
    Error : float
        The total error of the optimized solution.
    W : ndarray
        1D array containing the normalized
        weights of the evaluated reference days
    R : ndarray
        1D array the sampled random days
    """
        
    # Initiate the LP problem
    prob = pulp.LpProblem('Time_Series_Error_Problem', pulp.LpMinimize)

    # Specify the total error
    Error = pulp.LpVariable('Total_Error', lowBound=0, cat='Continuous')

    # Specify weights variables as LP variables
    Weights = pulp.LpVariable.dict('W', ('d' + str(day) for day in range(rdays.size)), lowBound=0, cat='Continuous')
    
    # Specify problem objective
    prob += Error, 'Total_Error'
    
    # Iterate over all time series
    for index, nparray in enumerate(series):
        
        # Calculate their A and L matrices
        A, L = matrices(nparray, rdays, nbins)
        
        # Specify the binerrors of the time series as LP variables
        binerrors = pulp.LpVariable.dict('ER_s' + str(index), ('b' + str(bin) for bin in range(nbins)), cat='Continuous')
        binerrorday = pulp.LpVariable.dict('ER_s' + str(index), ('b' + str(bin) + '_d' + str(day) for bin in range(nbins) for day in range(rdays.size)), cat='Continuous')
        
        for bin in range(nbins):
            for day in range(rdays.size):
                
                prob += binerrorday['b' + str(bin) + '_d' + str(day)] == Weights['d' + str(day)] * (np.divide(1, nslice) * A[day][bin])
            
            value = L[bin] - pulp.lpSum([binerrorday['b' + str(bin) + '_d' + str(day)] for day in range(rdays.size)])
            
            prob += value <= binerrors['b' + str(bin)]
            prob += (-1) * value <= binerrors['b' + str(bin)]
        
    # The sum of all the binerros equals the total error    
    prob += pulp.lpSum([variable for variable in prob.variables() if ('ER' in str(variable)) & ('d' not in str(variable))]) == Error
    
    # The sum of all the weights is equal to a year
    prob += pulp.lpSum(Weights[weight] for weight in Weights.keys()) == nslice
    
    # Solve the LP problem
    prob.solve()
    
    # Store the total error
    E = np.array([prob.objective.value()])
    
    # Store the results of the weight variables
    W = np.array([Weights[weight].varValue for weight in Weights.keys()])
    
    W = np.divide(W, nslice)
    
    return E, W, rdays


def nproblem(series, iterations, ndays=10, nbins=5, sslice=24, nmslice=[0, 365], seed=30):
    """
    Function that evaluates random samples of the given series.
    
    Parameters
    ----------
    series : ndarray
    iterations : int
        Number of samples to evaluate.       
    ndays : int, default 10
        Number of reference days that are randomly sampled and evaluated in the function.
    nbins : int, default 5
        Number of bins to use to split the series data.
    sslice : int, default 24
        Size of a slice. Specifies the size of the slice of a representative day.
    nmslice : list, default [0, 365]
        Slice range to include in the evaluation. Can be used to work with 
        series of different lengths.
    seed: int, default 30
        Seed of random number generator
        
    Returns
    -------
    E : ndarray
        1D array containg the total errors of the optimized solutions.
    W : ndarray
        2D array containing the normalized weights of the evaluated reference days
    R : ndarray
        2D array the sampled random days
    """
    
    E = np.zeros(shape=(iterations, 1), dtype=float)
    W = np.zeros(shape=(iterations, ndays), dtype=float)
    R = np.zeros(shape=(iterations, ndays), dtype=int)
    
    nslice = nmslice[1] - nmslice[0]
    
    # validate and covert passed list of series
    series = validation(series, sslice=sslice, nmslice=nmslice)
    
    # set random seed
    np.random.seed(seed)
    
    # evaluate problem n times
    for n in range(iterations):
        rdays = np.random.choice(np.arange(365), size=ndays, replace=False)
        E[n], W[n], R[n] = solver(series=series, rdays=rdays, nbins=nbins, sslice=sslice, nslice=nslice) 
    
    return E.ravel(), W, R


def nmproblem(series, iterations, ndays=10, nbins=10, sslice=24, nmslice=[0, 365], seed=30):
    """
    Function that evaluates random samples of the given series for a given 
    range of days and bins. Can be used to analyse trade-offs.
    
    Parameters
    ----------
    series : ndarray
    iterations : int
        Number of samples to evaluate for each day and bin combination.       
    ndays : int, default 10
        Range of number of days over which the function is evaluated
    nbins : int, default 5
        Range of number of bins over which the function is evaluated
    sslice : int, default 24
        Size of a slice. Specifies the size of the slice of a representative day.
    nmslice : list, default [0, 365]
        Slice range to include in the evaluation. Can be used to work with 
        series of different lengths.
    seed: int, default 30
        Seed of random number generator
        
    Returns
    -------
    Enm : ndarray
        2D containing the minimum erors of each day, bin combination.
    """    
    
    Enm = np.zeros(shape=(nbins, ndays), dtype=float)
    
    for n in range(nbins):
        for m in range(ndays):
            E, W, R = nproblem(series=series, iterations=iterations, ndays=m+1, nbins=n+2, sslice=sslice, nmslice=nmslice, seed=seed)
            
            index = np.where(E == E.min())[0]
            
            Enm[n, m] = E[index][0]
            
    return Enm


class EMAProblem():
    """
    Class to evaluate nmproblem as EM Workbench problem.

    Parameters
    ----------
    series : ndarray
    iterations : int
        Number of samples to evaluate for each day and bin combination.       
    ndays : int, default 10
        Range of number of days over which the function is evaluated
    nbins : int, default 5
        Range of number of bins over which the function is evaluated
    sslice : int, default 24
        Size of a slice. Specifies the size of the slice of a representative day.
    nmslice : list, default [0, 365]
        Slice range to include in the evaluation. Can be used to work with 
        series of different lengths.
    seed: int, default 30
        Seed of random number generator
        
    Returns
    -------
    Enm : ndarray
        2D containing the minimum erors of each day, bin combination.
    """

    def __init__(self, series, iterations, sslice=24, nmslice=[0, 365], init_seed=30):
        
        self.series = series
        self.iterations = iterations
        self.sslice = sslice
        self.nmslice = nmslice
        self.init_seed = init_seed
    
    def __call__(self, **kwargs):
    
        self.results = {}
        
        # kwargs setting
        for item in kwargs:

            if item == 'ndays':
                ndays = kwargs[item]

            elif item == 'nbins':
                nbins = kwargs[item]
                
            elif item == 'seed':
                seed = kwargs[item]

            else:
                raise KeyError(str([item]) + 'is not a model parameter')

        rseed = self.init_seed + seed
        
        E, W, R = nproblem(series=self.series, iterations=self.iterations, ndays=ndays, nbins=nbins,
                           sslice=self.sslice, nmslice=self.nmslice, seed=rseed)

        index = np.where(E == E.min())[0]
        
        self.results['error'] = E[index][0]
        self.results['weights'] = W[index][0]
        self.results['rdays'] = R[index][0]
 
        return self.results