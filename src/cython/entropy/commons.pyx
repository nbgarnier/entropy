#cython: language_level=3, boundscheck=False, cython.wraparound=False
# Cython wrappers for C functions
#
# 2018-04-10: enhanced memory mapping from C to Python
#       note: only some basic functions have been edited, changes have to be propagated
# 2019-01-24: enhanced tests, interface and documentation + cleaning
# 2020-02-23: now using memoryviews for all functions
#             http://docs.cython.org/en/latest/src/userguide/memoryviews.html#c-and-fortran-contiguous-memoryviews
# 2020-02-23: now using 1-d nd-array for mask (instead of 2-d nd-array)
# 2020-02-26: access to standard deviation of estimators (when stride>1), as well as nb_errors
# 2021-01-20: removed "bins" estimators
# 2021-12-02: multi-threaded entropy algorithm
# 2021-12-19: forked out of entropy_ann.pyx
# 2023-03-23: playing with memoryviews freeing for optimizations

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from cython cimport view                    # to export C pointer to ndarray, oldish version
from cython.view cimport array as cvarray   # to export C pointer to ndarray, newer version
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

@cython.boundscheck(False)
@cython.wraparound(False)
# @embedsignature(True)
def get_last_info(int verbosity=0):
    """ 
    returns informations from the last computation (and prints them on screen if verbosity>0)
    
    :param verbosity: an integer. If ==0 (default), then nothing is printed on the screen, but values are returned (useful for use in scripts).
    :returns: the following 8 values, in the following order
    
      - the standard deviations estimates of the last computed quantities
      - the number of errors encountered 
      - the effective number of points used, per realization, and total
      - the number of independent realizations used
      - the effective value of tau_Theiler (in x, and eventually in y)
      
    """
    cdef double std             = commons.last_std
    cdef double std2            = commons.last_std2
    cdef int    nb_errors_local = commons.nb_errors_local
    cdef int    nb_errors       = commons.nb_errors
    cdef int    N_eff_local     = commons.last_npts_eff_local
    cdef int    N_eff           = commons.last_samp.N_eff
    cdef int    N_realizations  = commons.last_samp.N_real
    cdef int    Theiler_x       = commons.last_samp.Theiler      
    cdef int    Theiler_y       = commons.samp_2d.last_Theiler_y
    
    if verbosity:
        print("from last function call:")
        print("- standard deviation(s):      %f, %f" %(std,std2))
        print("- nb of realizations:         %d" %N_realizations)
        print("- nb of errors encountered:   %d (total)" %nb_errors)
        print("- effective nb of points:     %d (per realization)" %N_eff_local) 
        print("                              %d (total)" %N_eff)
        print("- Theiler scale               %d (and %d if using a second direction)" %(Theiler_x, Theiler_y))
    return [std, std2, nb_errors, N_eff_local, N_eff, N_realizations, Theiler_x, Theiler_y]


def get_extra_info(int verbosity=0):
     """
     returns extra information from the last computation
     
     :param verbosity: an integer. If ==0 (default), then nothing is printed on the screen, but values are returned (useful in scripts).
     :returns: statistics on the processed input data (e.g.: increments of the input):

       - the standard deviation of the last data used (i.e., the std of the input, not of the output!)
       - the standard deviation of this standard deviation
       
     These maybe useful for, e.g., computations on increments with a given stride.
     """
     cdef double std       = commons.data_std
     cdef double std_std   = commons.data_std_std
     if verbosity:
        print("from last function call:")
        print("- standard deviation of processed input data  : %f" %std)
        print("- std of this very std                        : %f" %std_std) 
     return [std, std_std]


# 2019-01-28: OK
def choose_algorithm(int algo=1, int version=1):
     """
     selects the algorithms to use for computing all mutual informations (including partial mutual informations and TEs),
     and selects the algorithm to use to count neighbors.
     
     :param algo: the Kraskov-Stogbauer-Grassberger algorithm. Possible values are {1, 2, 1|2}) for (algo 1, algo 2, both algos). (default=1)
     :param version: counting algorithm version, possible values are {1, 2} for (legacy, mixed ANN) (default=1)
     :returns: no output
     
     about the "version" parameter:
       - 1 : legacy (NBG) : faster for small embedding dimensions (<=2)
       - 2 : mixed ANN    : faster for large embedding dimensions (>=4)
     """
     commons.ANN_choose_algorithm(algo|(version*0x0100))
     return


# 2022-05-17: OK
def set_verbosity(int level=1):
    """
    sets the verbosity level of the library
    
    :param verbosity: an integer that indicates the verbosity level of the library (default=1)
    :returns: no output
    
    verbosity can be:
      - <0 : no messages, even if an error is encountered (not recommended!)
      -  0 : messages only if an error is encountered
      -  1 : messages for errors and important warnings only (default)
      -  2 : more warnings and/or more details on the warnings
      -  ...
        
    """
    commons.lib_verbosity=level
    
def get_verbosity():
    """
    gets the current verbosity level of the library
    
    :param none:
    :returns: no output values, but a message indicating the verbosity level is printed in the console.
    
    verbosity level explanation:
      - <0 : no messages, even if an error is encountered (not recommended!)
      -  0 : messages only if an error is encountered
      -  1 : messages for errors and important warnings only (default)
      -  2 : more warnings and/or more details on the warnings
      -  ...
        
    """
    print("verbosity level", commons.lib_verbosity)


# 2022-05-17: OK
# 2022-11-30: little bug corrected (for "legacy" behavior)
def set_Theiler(int Theiler=4):
    """ 
    sets the way the library handles by default the Theiler prescription.
    this prescription is overiden if explicitly specified in a function call
    
    set_Theiler(Theiler='legacy'|'smart'|'random'|'adapted')
    
    :param Theiler: Theiler prescription. Possible values are {1, 2, 3, 4} or ("legacy", "smart", "random", "adapted") (default=4)
    :returns: no output values, but a message is printed in the console.
        
    The parameter "Theiler" indicates the Theiler prescription to follow:
      - 1 or "legacy"  : tau_Theiler=tau(=stride) + uniform sampling (thus localized in the dataset) (legacy)
      - 2 or "smart"   : tau_Theiler=max>=tau(=stride) + uniform sampling (covering the full dataset)
      - 3 or "random"  : tau_Theiler=tau(=stride) + random sampling
      - 4 or "adapted" : tau_Theiler>(or <)tau(=stride) 
        
    Depending on the Theiler prescription, the effective value of ''tau_Theiler'' can be smaller than tau(=stride) in order to satisfy the imposed N_eff. Use this with caution, for example by tracking the effectively selected ''tau_Theiler'' value with the function :any:`get_last_info`. 
    """
    if   ( (Theiler==1) or (Theiler=='legacy')  ): 
        commons.samp_default.type   =1
        commons.samp_default.N_eff  =-1   
        commons.samp_default.N_real =-1    
    elif ( (Theiler==2) or (Theiler=='smart')   ): commons.samp_default.type=2
    elif ( (Theiler==3) or (Theiler=='random')  ): commons.samp_default.type=3
    elif ( (Theiler==4) or (Theiler=='adapted') ): commons.samp_default.type=4
    else: print("please provide a valid prescription (int or string) (see help)")
    commons.samp_default.Theiler=-commons.samp_default.type # 2022-05-24: important because most looked at
    print("now using Theiler prescription %d" %commons.samp_default.type)
    return


# 2022-05-23: OK
# 2023-11-17: N_real==1 now accepted
# 2024-09-18: N_eff and N_real now also set to default values for Theiler==1 
def set_sampling(int Theiler=4, int N_eff=4096, int N_real=10):
    """ set_sampling(Theiler='adapted', N_eff=4096, N_real=10)
    
    sets the way the library handles by default the Theiler prescription, the number N_eff of effective points and the number N_real of realizations.
    These values are overiden if explicitly specified in a function call. 
    
    :param Theiler: prescription (default='adapted'), see function :any:`set_Theiler` for details on possible options.
    :param N_eff: number of points to consider in the statistics (default=4096).
    :param N_real: number of realizations to consider (default=10).
    :returns: no output.
    
    parameters N_eff and N_real are overriden if Theiler=='legacy'
    
    You can check what are the current default values with the function :any:`get_sampling`. You can also examine what were the values used in the last computation with the function :any:`get_last_sampling`. See :any:`input_parameters`
    """    
    set_Theiler(Theiler)
    if (commons.samp_default.type>0): # 2024-09-18: >1 not legacy replaced by larger condition
        if (N_eff>1)  : commons.samp_default.N_eff=N_eff
        if (N_real>0) : commons.samp_default.N_real=N_real
    return


def get_sampling(verbosity=1):
    """
    prints the default values of sampling parameters used in all functions.
    
    :param verbosity: an integer in {0,1} (default=1)
    :returns: 4 values described below. If verbosity>0, a human-readable message expliciting these 4 values is also printed in the console.
    
    Returned are the following 4 values, in the following order: 
      - the default type of Theiler prescription
      - the default Theiler scale
      - the default effective number of points used in a single realization
      - the default number of realizations used.

    see :any:`set_sampling` to change these values and :any:`input_parameters` for their meaning.
    see :any:`get_last_sampling` to get the last values instead (they may be different than default values).
    """
    if (verbosity>0):
        print("Theiler prescription :", commons.samp_default.type, end="")
        if   (commons.samp_default.type==1): print("(legacy)")
        elif (commons.samp_default.type==2): print("(smart)")
        elif (commons.samp_default.type==3): print("(random)")
        elif (commons.samp_default.type==4): print("(adapted)")
        else:                                print("unknown?")
        print("Theiler scale        :", commons.samp_default.Theiler)
        print("N_eff                :", commons.samp_default.N_eff)
        print("N_realizations       :", commons.samp_default.N_real)
    return [commons.samp_default.type, commons.samp_default.Theiler, commons.samp_default.N_eff, commons.samp_default.N_real]
    
def get_last_sampling(int verbosity=0):
    """
    returns full set of information on the sampling used in the last computation and prints them in the console if verbosity>0
    (note that these values may differ from default values, as returned by :any:`get_sampling`)
    
    :param verbosity: an integer in {0,1} (default=0). If verbosity>0, a human-readable message expliciting the 7 returned values is printed in the console.
    :returns: the following 7 values, in the following order: 
      - the type of Theiler prescription ( 1 integer)
      - the Theiler scale used, and its maximal value given the other parameters (2 integers)
      - the effective number of points used in a single realization, and its maximal value (2 integers)
      - the number of realizations used, and its maximal value (2 integers)
    """
    if (verbosity>0):
        print("from last function call:")
        print("- Theiler prescription :", commons.last_samp.type)
        print("- Theiler value        :", commons.last_samp.Theiler, "max :", commons.last_samp.Theiler_max)
        print("- N_eff                :", commons.last_samp.N_eff,   "max :", commons.last_samp.N_eff_max)
        print("- N_realizations       :", commons.last_samp.N_real,  "max :", commons.last_samp.N_real_max)
    return [commons.last_samp.type, commons.last_samp.Theiler, commons.last_samp.Theiler_max, commons.last_samp.N_eff, commons.last_samp.N_eff_max, commons.last_samp.N_real, commons.last_samp.N_real_max]

    
# 2022-01-12: OK
def set_Theiler_2d(int Theiler=2):
    """
    selects the 2-d Theiler prescription to use for sampling images.
    
    :param Theiler: Theiler prescription. Possible values: are {1, 2, 4} as explicited below.
    :returns: no output
    
    About possible values:
      - 1 or "minimal" : tau_Theiler is selected in each direction as in 1-d (troublesome if one of the stride is much smaller than the other one)
      - 2 or "maximal" : tau_Theiler is selected as the max of (stride_x, stride_y) (possibly too small by a factor sqrt(2))
      - 4 or "optimal" : tau_Theiler is selected as sqrt(stride_x^2 + stride_y^2) (rounded-up for max safety)
    (default=2)
    """
    if   ( (Theiler==1) or (Theiler=='minimal') ): commons.samp_2d.type=1
    elif ( (Theiler==2) or (Theiler=='maximal') ): commons.samp_2d.type=2
    elif ( (Theiler==4) or (Theiler=='optimal') ): commons.samp_2d.type=4
    else: print("please provide a valid prescription (int or string) (see help)")
    print("now using 2-d prescription #%d" %commons.samp_2d.type)
    return


# 2022-01-14: OK
def get_Theiler_2d():
     """
     returns the Theiler prescription currently selected.
     (see the function :any:`set_Theiler_2d` for details)
     
     :param none:
     :returns: the default Theiler prescription currently used.
     """
     return(commons.samp_2d.type)


# 2021-12-02
def multithreading(do_what="info", int nb_cores=0):
     """ multithreading(do_what=["info", "auto", "single"], n_cores=0)

     Selects the multi-threading schemes and eventually sets the number of threads to use
     
     :param do_what: either "info" or "auto" or "single", see below.
     :param nb_cores: an integer; if specified and positive, then the provided number nb_cores of threads will be used. (default=0 for "auto") 
     :returns: no output.
     
     The parameter do_what can be chosen as follows:
        - "info": nothing is done, but informations on the current multithreading state are displayed.
        - "auto": then the optimal (self-adapted to largest) number of threads will be used) (default).
        - "single": then the algorithms will run single-threaded (no multithreading) 
     
     if nb_cores (a positive number) is specified, then the provided number nb_cores of threads will be used.
     """
     
     if (do_what=="info"): 
        print("currently using %d out of %d cores available"
                    %(commons.get_cores_number(0x0001), commons.get_cores_number(0x0010)), end="")
        if (commons.get_cores_number(0x0001)==-1): 
                print(" (-1 means largest number available, so %d here)" %commons.get_cores_number(0x0010))
        else:   print()
        return
     elif (do_what=="auto"):                    # autodetect/automatic (default)
        commons.set_multithreading_state(2)
        commons.set_cores_number(-1)
     elif (do_what=="single"):                  # single threading
        commons.set_multithreading_state(0)     # no multithreading
        commons.set_cores_number(1)
     elif ( isinstance(do_what, (int, PNP.int)) and (do_what>0) ):      # imposed nb of threads
# see https://stackoverflow.com/questions/21851985/difference-between-np-int-np-int-int-and-np-int-t-in-cython
        commons.set_multithreading_state(1) 
        commons.set_cores_number(do_what)
     else: raise ValueError("invalid parameter value")
     return


# 2021-12-20
def get_threads_number():
     """
     returns the current number of threads.
     
     :param none:
     :returns: an integer, the current number of threads used.
      
     """
     
     if (commons.USE_PTHREAD>0): return(commons.get_cores_number(0x0001))
     return(0)



#####################################################################################
# 2023-02-03: following function may prove usefull one day
#             in function 'surrogate()', another approach was taken.
#             so I comment this out
#####################################################################################
cdef pointer_to_numpy_array(void * ptr, CNP.npy_intp size0, CNP.npy_intp size1):
    """
    converts C double pointer a numpy float array of shape (size0, size1).
    The memory will be freed as soon as the ndarray is deallocated.
    """
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(CNP.ndarray arr, int flags)
    cdef CNP.npy_intp[2] my_shape = [size0, size1]
    cdef CNP.ndarray[dtype=double, ndim=2] arr = CNP.PyArray_SimpleNewFromData(2, my_shape, CNP.NPY_DOUBLE, ptr)
    PyArray_ENABLEFLAGS(arr, PNP.NPY_OWNDATA)
    return arr



#####################################################################################
# 2023-02-06: following function may prove usefull one day
#             but is is not working yet
#####################################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
def surrogate_WIP( double[:, ::1] x):
    """ 
    xs = surrogate(x) : creates a surrogate version of (possibly multi-dimensionial) data x
    by shuffling points in time, while retaining the joint-coordinates
    
    :meta private:
    """
    cdef int npts=x.shape[1], nx=x.shape[0]
    if (npts<nx):  raise ValueError("please transpose x")
    
## the following lines give direct access to memory, but the drawback is that memory is not freed
## so there is an extra line to express explicitly how to free it   
#    cdef double *z = commons.create_surrogate(&x[0,0], npts, nx)
##    cdef double[:,:] my_array = <double[:nx,:npts]> z  # I'm not sure this works with view.array, next line may be better
#    cdef view.array my_array = <double[:nx,:npts]> z    # this should be the same as line, above, but validated by cython dev
#    my_array.callback_free_data = free                  # important: define a function that can deallocate the data
#    return PNP.asarray(my_array)
## https://stackoverflow.com/questions/25102409/c-malloc-array-pointer-return-in-cython
## https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#cython-arrays

# another way of doing it:
    cdef double *z = commons.create_surrogate(&x[0,0], npts, nx)    
    return(pointer_to_numpy_array(z, nx, npts))



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef surrogate( double[:, ::1] x, int method=0, int N_steps=7):
    """
    creates a surrogate version of (possibly multi-dimensionial) data x
    
    :param method: an integer to indicate which type of surrogate to create (see below)
    :param N_steps: an integer to indicate how many steps to use for improved surrogates (method 4 only)
    :returns: an nd-array with the same dimmensionality as the input data x
    
    The method parameter can be:  
      |  0 : by shuffling points in time, while retaining the joint-coordinates
      |  1 : randomize phase with unwindowed Fourier transform (uFt)
      |  2 : randomize phase with windowed Fourier transform (wFT) (buggy!)
      |  3 : randomize phase with Gaussian null hypothesis and Ft (aaFT)
      |  4 : improved surrogate (same PDF and PSD), using N_steps
      |  5 : creates a Gaussian version of x, with same PSD and dependences

    """
    cdef int npts=x.shape[1], nx=x.shape[0], m
    if (npts<nx):  raise ValueError("please transpose x")
    if ((method<0) or (method>5)): raise ValueError("choose a valid method")
    
    zout = PNP.zeros((nx, npts), dtype=float)
    cdef double[:, :] z = zout
    memcpy(&z[0,0], &x[0,0], nx*npts*sizeof(double))
    if (method==0):    commons.shuffle_data  (&z[0,0], npts, nx) 
    elif (method==1):  commons.surrogate_uFt (&z[0,0], npts, nx) 
    elif (method==2):  commons.surrogate_wFt (&z[0,0], npts, nx) 
    elif (method==3):  commons.surrogate_aaFt(&z[0,0], npts, nx)
    elif (method==4):  commons.surrogate_improved(&z[0,0], npts, nx, N_steps)
    elif (method==5):  
        means = PNP.mean(x, axis=1)
        stds  = PNP.std (x, axis=1)
        commons.Gaussianize(&z[0,0], npts, nx)
        if (nx>1): 
            for m in PNP.arange(nx):    z[m,:] = z[m,:]*stds[m] + means[m]
        else: 
            z = z*stds + means
    return(zout)



@cython.boundscheck(False)
@cython.wraparound(False)
def mask_finite_C(double[:, ::1] x):
    """ 
    mask=mask_finite(x) : build a mask corresponding to non-NaN values in the data x
    
    C-version, experimental.

    :param x: data (possibly multi-dimensional)
    :returns: a mask corresponding only to finite points in data

    :meta private:
    """
  
    cdef int npts=x.shape[1], nx=x.shape[0], ratou
    if (npts < nx): raise ValueError("please transpose x")
          
    cdef CNP.ndarray[dtype=char, ndim=1] mask = PNP.ones(npts, dtype='i1') # type 'i1' is 'char'
    ratou = commons.finite_mask(&x[0,0], npts, nx, &mask[0])
#    print("I found %d non finite points" %ratou)
    return mask
