"""
setup.py for package "entropy"
(using Cython to interface the C library) 
2014-09-30
2015-10-06
2018-04-10
2019-10-14 now using python3
2021-12-20 source code split into several files
2022-11-25 now using build instead of setup
"""
from setuptools import setup, Extension
#from distutils.core import setup, Extension
#from Cython.Distutils import build_ext 
from Cython.Build import cythonize

# to have an annotated html:
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
DO_ANNOTATE = False

# Third-party modules - we depend on numpy for everything
import numpy
import sys
#import os.path
import os

# Obtain the numpy include directory:
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
# another method to get the include directory :
#numpy_include='%s/lib/python%s/site-packages/numpy/core/include'\
#         %(os.path.normpath(sys.exec_prefix),sys.version[:3])
print("using NumPy from : ", numpy_include)

import subprocess 
#LIBS_PYTHON = subprocess.run(["python3-config","--libs"], capture_output=True).stdout # python3.7
#LIBS_PYTHON = subprocess.run(["python3-config","--libs"], stdout=subprocess.PIPE).stdout # python3.5
#LIBS_PYTHON = LIBS_PYTHON.rstrip().decode().split()
#LDFLAGS_PYTHON = subprocess.run(["python3-config", "--ldflags"], capture_output=True).stdout # python3.7
LDFLAGS_PYTHON = subprocess.run(["python3-config", "--ldflags"], stdout=subprocess.PIPE).stdout # python3.5
LDFLAGS_PYTHON = LDFLAGS_PYTHON.rstrip().decode().split()
#print("LIBS_PYTHON", LIBS_PYTHON)
print("LDFLAGS_PYTHON", LDFLAGS_PYTHON + ["-Wl,-undefined,error"]) 


print(sys.version)
import platform
if (platform.system()=='Darwin'):
		LIBS = ['c++']      # 2018-11-27: stdc++ replaced by c++ (for macos clang at least)
		LDFLAGS_PYTHON = ['-mmacosx-version-min=12'] # 2021-02-15: replaced 10.12 by 10.15 
		                                             # 2022-11-26: replaced 10.15 by 12 (Monterey)
else:
		LIBS = ['stdc++', 'm']
		
print(LIBS)

INC_DIR  = ['.', '../include', '../../include', './entropy', numpy_include]
LIBS_DIR = ['../../lib', '/usr/lib', '/usr/local/lib'] # '/opt/local/lib'
LIBS     = ['entropy', 'ANN', 'gsl', 'gslcblas', 'fftw3', 'pthread'] + LIBS #'m', # 2020-07-22: moved LIBS at the end (for stdc++ on linux)

# 2018-04-11: 'python' replaced by 'python2.7'
# 2019-10-14: 'python2.7' replaced by 'python3.7m', and then by auto

# 2022-02-14: for debug:
#import Cython.Compiler.Options
#Cython.Compiler.Options.annotate = True

# 2021-12-19: testing with several modules => useless for ELF libraries which do not share symbols
# 2021-12-20: so only 1 module (named "entropy" below) is used
entropy_module = Extension("entropy",
				sources = ['entropy/entropy.pyx'],
#				include_path= ["entropy/"],
                include_dirs = INC_DIR,
                libraries    = LIBS,
                library_dirs = LIBS_DIR,
                extra_link_args=LDFLAGS_PYTHON,
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                )

commons_module = Extension("commons",
				sources = ['entropy/commons.pyx', 'entropy/commons.pxd'],
                include_dirs = INC_DIR,
                libraries    = LIBS,
                library_dirs = LIBS_DIR,
                extra_link_args=LDFLAGS_PYTHON,
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                )

computes_module = Extension("compute",
				sources = ['entropy/computes.pyx', 'entropy/computes.pxd'],
#				 cython_opts = '',
                include_dirs = INC_DIR,
                libraries    = LIBS,
                library_dirs = LIBS_DIR,
#                   extra_compile_args=["-mmacosx-version-min=10.9"],
#                  language = "c++",
#                  extra_link_args=['-framework', 'CoreFoundation'],
# 2015/10/06, NG, line above added for macos. seems useless.
# 2018-04-10, line added back, and then removed.
                extra_link_args=LDFLAGS_PYTHON,
#                   extra_link_args=["-Wl,-undefined,error"]
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] # cf https://github.com/scipy/scipy/issues/5889
                )

others_module = Extension("others",
				sources = ['entropy/others.pyx', 'entropy/others.pxd'],
                include_dirs = INC_DIR,
                libraries    = LIBS,
                library_dirs = LIBS_DIR,
                extra_link_args=LDFLAGS_PYTHON,
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                )

#entropy_tools_module = Extension('tools')
                       
setup(name = 'entropy',
      version = '3.3.1', 
#      date ='2023-03-01',
      description = "Information Theory tools and entropies for multi-scale analysis of continuous signals",
      author      = "Nicolas B. Garnier",
      author_email= "nicolas.garnier@ens-lyon.fr",
      include_dirs = [numpy.get_include()],         # New line 2014/09/30
#      zip_safe=False,  # 2022-11-25, to work with cimport for pxd files when using them from a dependent package
                        # 2023-02-14 : line above commented
#	  cmdclass = {'build_ext': build_ext},			
	  ext_package='entropy',
#	  package_dir={'', 'entropy', 'tools'},
#      py_modules  = ['tools.tools', 'tools.masks'],
      py_modules  = ['entropy.tools', 'entropy.masks'],
#      ext_modules = [entropy_module],  # good version  
      ext_modules = cythonize([entropy_module], annotate=DO_ANNOTATE), # tried 2019-10-14 to replace line above
#	  packages = ['entropy']
	)

