#cython: language_level=3
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
# 2021-12-19: renamed "entropy_ann.pyx" as this file "entropy.pyx" 
# 2021-12-20: using includes

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import  cython
import  numpy as PNP # shoud be useless here
cimport numpy as CNP

# on initialise NumPy : # necessaire... 
CNP.import_array()

# on importe les definitions du C (fichiers .pxd) :
cimport entropy.commons
cimport entropy.computes as entropy
cimport entropy.others

include "commons.pyx"
include "computes.pyx"
include "others.pyx"

