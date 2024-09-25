# tools (optimized version)
#
# 2024-09-24: new cython version, forked from python version tools.py
#cython: language_level=3
import  cython
import  numpy as PNP # shoud be useless here
cimport numpy as CNP
from time import time

# on initialise NumPy : # necessaire... 
CNP.import_array()

# on importe les definitions du C (fichiers .pxd) :
cimport entropy.tools

from entropy.entropy import get_last_info
from entropy.entropy import get_last_sampling

# this usefull function insures that the time dimension is the first dimension
# if we are dealing with a 1-d ndarray, it is cast into a 2-d array
# 2023-10-26: now also insures that data is C-contiguous
def reorder(x):
    """
    makes any nd-array compatible with any function of the code for temporal-like signals
    
    :param x: any nd-array
    :returns: a well aligned and ordered nd-array containing the same data as input x
    """
    if (x.ndim==1):    # ndarray of dim==1
        x=x.reshape((1,x.size))
    elif (x.ndim==2):  # regular matrix (for multivariate 1-d data)
        if (x.shape[0]>x.shape[1]):
            x=x.transpose()
    else:
        print("please use reorder_2d for multivariate images")
    if x.flags['C_CONTIGUOUS']: return x
    else:                       return x.copy()
    

# this function is for (possibly multivariate) images
# 2023-10-26: now also insures that data is C-contiguous
def reorder_2d(x, nx=-1, ny=-1, d=-1):
    """
    makes any nd-array compatible with any function of the 2d code (images)
    """
    if (x.ndim==1):    # ndarray of dim==1
        if ((nx>0) and (ny>0)):
            x=PNP.reshape(x,(nx,-1),'F')
    elif (x.ndim==2):  # regular matrix (image)
        print("nothing to do")
    elif (x.ndim==3):  # tensor for multivariate image
        x=PNP.reshape(x,(x.shape(0),x.shape(1)),'F')
    else:
        print("order>3 not supported")
    if x.flags['C_CONTIGUOUS']: return x
    else:                       return x.copy()


## 2020-02-22: compatible with new convention of cython code (ndim, npts)
## 2020-02-23: this embedding is causal, checked OK
def embed_python(x, m=1, stride=1, i_window=0):
    """
    (time-)embeds an nd-array x (possibly multi-dimensional)
    
    note: this function is only here for test purposes; it is absolutely not optimized in any way.
    you should use the function "embed" instead.
    
    :param x: signal (NumPy array with ndim=2, time along second dimension)
    :param  m: embedding dimension (default=1)
    :param stride: distance between successive points (default=1)
    :param i_window: returns the (i_window)th set (0<=i_window<stride)
    :returns: an nd-array with the requested (time-)embeded version of input data x
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
    x_new  = PNP.zeros((n, npts_new))
    
#    for i_window in numpy.arange(n_windows):  # loop over independant windows
    for i in PNP.arange(npts_new):  # loop on time in 1 window
            for d in PNP.arange(mx):  # loop on existing dimensions in x
                for l in PNP.arange(m):  # loop on embedding
                    x_new[d + l*mx, i] = x[d, i_window + n_windows*i + stride*(m-1-l)];
                    
    return x_new

@cython.boundscheck(False)
@cython.wraparound(False)
def embed(double[:, ::1] x, int n_embed=1, int stride=1, int i_window=0, int n_embed_max=-1):
    """ y = embed(x, [n_embed=1, stride=1, i_window=0])
    
    causal (time-)embed an nd-array x (possibly multi-dimensional), (optimized function much faster then the _python version)

    :param x: signal (NumPy array with ndim=2, time along second dimension)
    :param n_embed: embedding dimension (default=1)
    :param stride: distance between successive points (default=1)
    :param i_window: returns the (i_window)th set (0<=i_window<stride)
    :param n_embed_max: max embedding dimension (for causal time reference); if -1, then uses n_embed (default=-1)
    :returns: an nd-array with the requested (time-)embeded version of input data x
    """

    cdef int npts=x.shape[1], nx=x.shape[0]
    if (npts<nx):  raise ValueError("please transpose x")
    if (i_window>=stride): raise ValueError("i_window must be smaller than stride")
    
    cdef int pp = n_embed if n_embed_max<=0 else n_embed_max
    cdef int nb_pts_new = entropy.tools.Theiler_nb_pts_new(npts, stride, pp)
#    cdef double[:, ::1] output = PNP.zeros((nx*n_embed,nb_pts_new), dtype=float) # works fine, but below is faster? (2021-02-05)
    cdef CNP.ndarray[dtype=double, ndim=2] output = PNP.zeros((nx*n_embed,nb_pts_new), dtype=float) 

    entropy.tools.time_embed(&x[0,0], &output[0,0], npts, nb_pts_new, nx, n_embed, pp, stride, i_window, stride)
    return PNP.asarray(output)


@cython.boundscheck(False)
@cython.wraparound(False)
def crop(double[:, ::1] x, int npts_new, int i_window=0):
    """ y = crop(x, npts_new, [i_window=0])
    
    crop an nd-array x (possibly multi-dimensional) in time (faster than Python)

    :param x: signal (NumPy array with ndim=2, time along second dimension)
    :param npts_new: new size in time
    :param i_window: starting point in time (default=0)
    :returns: an nd-array with the requested version of input data x
    """

    cdef int npts=x.shape[1], nx=x.shape[0]
    if (npts<nx):  raise ValueError("please transpose x")
    if (i_window+npts_new>npts): raise ValueError("i_window (%d) + npts_new (%d) is larger than npts (%d)" %(i_window, npts_new, npts))
    
    cdef CNP.ndarray[dtype=double, ndim=2] output = PNP.zeros((nx, npts_new), dtype=float) 

    entropy.tools.crop_array(&x[0,0], &output[0,0], npts, nx, npts_new, i_window)
    return PNP.asarray(output)




#%% 2023-10-26, added to the library after thoughful testing
def compute_over_scales(func, tau_set, *args, verbosity_timing=1, get_samplings=0, **kwargs):
    """
    runs iteratively an estimation over a range of time-scales/stride values
    
    :param func: (full) name of the function to run
    :param tau_set: 1-d numpy array containing the set of values for stride (time-scales)
    :param verbosity_timing: 0 for no output, or 1,2 or more for more and more detailed output
    :param get_samplings: 1 for extra returned array, with samplings parameters used for each stride
    :param any parameter to pass to the function: is accepted, with the same syntax as usual (e.g.: x, y, k=5, ...)
    
    :returns: 2 or 3 nd-arrays, each having their last dimension equal to tau_set.size (see below).
    
    example::
        
        # assuming you have data x and y available and ready for analysis
        
        tau_set = numpy.arange(20)      # a set of (time-)scales to examine
        MI, MI_std = compute_over_scales(entropy.compute_MI, tau_set, x, y, N_eff=1973)
        
        plt.errorbar(tau_set, MI, yerr=MI_std)
        
    About returned values:
      - the first returned array contains result(s) as a function of stride. If the function func returns more than one value, then this first nd-array will have more than one dimension.
      - the second returned array contains an estimator of the std, as a function of stride.
      - the third returned array (returned only if the parameter get_samplings is set to 1) contains all sampling parameters used for each estimation. This is usefull to track the value of N_eff or tau-Theiler used in the estimation as a function of the (time-)scale. See :any:`get_sampling` for a list of returned values, and how they are ordered.
    """
    
    test = func(*args, stride=tau_set[0], **kwargs)
    if isinstance(test, list):      # multiple returned values
        N_results = len(test)
        test_2 = test[0]
        if isinstance(test_2, CNP.ndarray): 
 #           print("function returns a ndarray of shape and size", test_2.shape, test_2.size)
            res = PNP.zeros((test_2.size, N_results, tau_set.size), dtype=float) # quantity
        elif isinstance(test_2, float):
 #           print("function returns a float")
            res = PNP.zeros((N_results, tau_set.size), dtype=float) # quantity
    else:                           # single retuned value
        res = PNP.zeros(tau_set.size, dtype=float)   # quantity
    
    res_std = PNP.zeros(tau_set.shape, dtype=float) # estimated errorbars
    if get_samplings: samplings = PNP.zeros((7, tau_set.size), dtype=float) # sampling parameters
        
    for i_tau in enumerate(tau_set):
        i = i_tau[0]
        if (verbosity_timing>1):   print(i+1, "/", tau_set.size, "tau =", i_tau[1], end='')
        elif (verbosity_timing>0): print(".", end='')
   
        time1=time()
        res[...,i] = PNP.atleast_2d( func(*args, stride=i_tau[1], **kwargs) )
        res_std[i] = get_last_info()[0]
        if get_samplings: samplings[...,i] = get_last_sampling()
        time2=time()
        if (verbosity_timing>2): print(" ->", res[...,i], "\t elapsed time: %f" %(time2-time1))
        if (verbosity_timing>1): print()
    if (verbosity_timing==1): print()
    
    if get_samplings: return res, res_std, samplings
    else:             return res, res_std

