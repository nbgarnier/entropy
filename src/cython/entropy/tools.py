import numpy

# this usefull function insures that the time dimension is the first dimension
# if we are dealing with a 1-d ndarray, it is cast into a 2-d array
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
    return x
    

# this function is for (possibly multivariate) images
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
    return x


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
    