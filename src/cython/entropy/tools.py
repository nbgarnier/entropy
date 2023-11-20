import numpy
from time import time
from entropy.entropy import get_last_info
from entropy.entropy import get_last_sampling

# this usefull function insures that the time dimension is the first dimension
# if we are dealing with a 1-d ndarray, it is cast into a 2-d array
# 2023-10-26: now also insures that data is C-contiguous
def reorder(x):
    """
    Function to make any ndarray compatible with any function of the code for temporal-like signals
    """
    if (x.ndim==1):    # ndarray of dim==1
        x=x.reshape((1,x.size))
    elif (x.ndim==2):  # regular matrix (for multivariate 1-d data)
        if (x.shape[0]>x.shape[1]):
            x=x.transpose()
    else:
        print("please use reorder_2d for multivariate images")
    if x.flags['C_CONTIGUOUS']: return x
    else:                      return x.copy()
    

# this function is for (possibly multivariate) images
# 2023-10-26: now also insures that data is C-contiguous
def reorder_2(x, nx=-1, ny=-1, d=-1):
    """
    Function to make any ndarray compatible with any function of the 2d code (images)
    """
    if (x.ndim==1):    # ndarray of dim==1
        if ((nx>0) and (ny>0)):
            x=numpy.reshape(x,(nx,-1),'F')
    elif (x.ndim==2):  # regular matrix (image)
        print("nothing to do")
    elif (x.ndim==3):  # tensor for multivariate image
        x=numpy.reshape(x,(x.shape(0),x.shape(1)),'F')
    else:
        print("order>3 not supported")
    if x.flags['C_CONTIGUOUS']: return x
    else:                       return x.copy()


## 2020-02-22: compatible with new convention of cython code (ndim, npts)
## 2020-02-23: this embedding is causal, checked OK
def embed(x, m=1, stride=1, i_window=0):
    """
    Function to time-embed a vector x (possibly multi-dimensional)
    m        : embedding dimension (default=1)
    stride   : distance between successive points (default=1)
    i_window : returns the (i_window)th set (0<=i_window<stride)
    """
    
    x    = reorder(x)
    npts = x.shape[1]
    mx   = x.shape[0]

    # Theiler requirements:
    npts_new  = (npts-npts%stride)//stride - (m-1); # size of a single dataset
    n_windows = stride; # there are n_window different datasets to work with

    if (i_window>=n_windows):
        raise ValueError("i_window must be less than stride!")
    
    n=mx*m
    x_new  = numpy.zeros((n, npts_new))
    
#    for i_window in numpy.arange(n_windows):  # loop over independant windows
    for i in numpy.arange(npts_new):  # loop on time in 1 window
            for d in numpy.arange(mx):  # loop on existing dimensions in x
                for l in numpy.arange(m):  # loop on embedding
                    x_new[d + l*mx, i] = x[d, i_window + n_windows*i + stride*(m-1-l)];
                    
    return x_new


#%% 2023-10-26, added to the library after thoughful testing
def compute_over_scales(func, tau_set, *args, verbosity_timing=1, get_samplings=0, **kwargs):
    """
    Function to run iteratively an estiation over a range of time-scales/stride values
    func             : (full) name of the function to run
    tau_set          : 1-d numpy array containing the set of values for stride (time-scales)
    verbosity_timing : 0 for no output, or 1,2 or more for more and more detailed output
    get_samplings    : 1 for extra returned array, with samplings parameters used for each stride
    all parameters to pass to the function are accepted, with the same syntax as usual (e.g.: x, y, k=5, ...)
    
    returned are 2 numpy arrays of last dimension tau_set.size: 
    - the first one contains result(s) as a function of stride,
    - the second one contains an estimator of the std, as a function of stride
    """
    test = func(*args, **kwargs, stride=2)
    if isinstance(test, list):      # multiple returned values
        N_results = len(test)
        test_2 = test[0]
        if isinstance(test_2, numpy.ndarray): 
 #           print("function returns a ndarray of shape and size", test_2.shape, test_2.size)
            res = numpy.zeros((test_2.size, N_results, tau_set.size), dtype=float) # quantity
        elif isinstance(test_2, float):
 #           print("function returns a float")
            res = numpy.zeros((N_results, tau_set.size), dtype=float) # quantity
    else:                           # single retuned value
        res = numpy.zeros(tau_set.size, dtype=float)   # quantity
    
    res_std = numpy.zeros(tau_set.shape, dtype=float) # estimated errorbars
    if get_samplings: samplings = numpy.zeros((7, tau_set.size), dtype=float) # sampling parameters
        
    for i_tau in enumerate(tau_set):
        i = i_tau[0]
        if (verbosity_timing>1):   print(i+1, "/", tau_set.size, "tau =", i_tau[1], end='')
        elif (verbosity_timing>0): print(".", end='')
   
        time1=time()
        res[...,i] = numpy.atleast_2d( func(*args, stride=i_tau[1], **kwargs) )
        res_std[i] = get_last_info()[0]
        if get_samplings: samplings[...,i] = get_last_sampling()
        time2=time()
        if (verbosity_timing>2): print(" ->", res[...,i], "\t elapsed time: %f" %(time2-time1))
        if (verbosity_timing>1): print()
    if (verbosity_timing==1): print()
    
    if get_samplings: return res, res_std, samplings
    return res, res_std

