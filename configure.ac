# -*- Autoconf script for entropy -*-
# N.B.Garnier, 2023-09-22
#################################################################

AC_PREREQ([2.71])
AC_INIT([entropy], [4.1.0], [nicolas.garnier@ens-lyon.fr])
AC_CONFIG_SRCDIR([src/entropy_ann.c])
AC_CONFIG_HEADERS([config.h])

#################################################################
# check for platform / system type:
#################################################################
AC_CANONICAL_HOST
#AC_CANONICAL_BUILD
# m4_case (host_os, linux-gnu, if-value-1, [value-2], [if-value-2], ..., [default])
# Detect the target system
AC_MSG_CHECKING([the build platform])
PLATFORM=unknown
MEX_LDEXTENSION=unknown
MEX_DIRNAME=unknown

case "${host_os}" in
    linux*)
        PLATFORM=[LINUX]
        MAKELIB="ar ruv"
        MEX_LDEXTENSION=[.mexa64]
        MEX_DIRNAME=[glnxa86]
        AC_MSG_RESULT([LINUX])
        ;;
    cygwin*|mingw*)
        PLATFORM=[WIN64]
        MAKELIB="ar ruv"
        MEX_LDEXTENSION=[.mexw64]
        MEX_DIRNAME=[maci86]
        AC_MSG_RESULT([WIN64])
        ;;
    darwin*)
        PLATFORM=[MACOS]
        AC_MSG_RESULT([MACOS])
        MAKELIB="libtool -arch_only $host_cpu -static -o"
        case "${host_cpu}" in
            x86_64*)
                MEX_LDEXTENSION=[.mexmaci64]
                MEX_DIRNAME=[maci64]
                ;;
            aarch64*)
                MEX_LDEXTENSION=[.mexmaca64]
                MEX_DIRNAME=[maca64]
                ;;
            *)
                AC_MSG_ERROR(["MacOS CPU $host_cpu is not supported"])
                ;;
            esac
        ;;
    *)
        AC_MSG_ERROR(["OS $host_os is not supported"])
        ;;
esac

#################################################################
# compiler flags
#################################################################
: ${CXXFLAGS="-O3 -fPIC -pthread"}
: ${CFLAGS="-O3 -fPIC -std=c99 -pthread"}
my_CPPFLAGS="-pedantic -Wall -Wextra -Wshadow -Wpointer-arith -Wundef -Wunreachable-code"

#################################################################
# Checks for programs.
#################################################################
AC_PROG_CXX([g++ cc++ clang++])
AC_PROG_CC([gcc cc clang])
AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_CHECK_PROGS(libtool, [libtool glibtool])   # libtool for Macos (clang) and linux, glibtool for homebrew on Macos
AC_CHECK_PROGS(PYTHON, [python python3], none)
AC_CHECK_PROGS(CYTHON, [cython cython3], none)
AC_CHECK_PROGS(TAR, tar)
AC_PATH_PROG(MATLAB, matlab, none)
#AC_MSG_NOTICE(["MATLAB" variable contains $MATLAB])
AS_IF([test "$MATLAB" = none], 
    [AC_MSG_NOTICE([no matlab installation found, you can specify Matlab main path using MATLAB=path])
    ],
    [
#    MATLAB=esyscmd([dirname [$MATLAB]])
    AC_MSG_NOTICE([using "MATLAB" from $MATLAB])
    ])
#AC_MSG_NOTICE(["MATLAB" variable contains $MATLAB])

#################################################################
# Checks for libraries:
#################################################################
AC_CHECK_LIB(m, log)
AC_CHECK_LIB(gsl, gsl_sf_psi, [], [AC_MSG_ERROR([please install libgsl-dev on your system (sudo apt install libgsl-dev)])])
AC_CHECK_LIB(fftw3, fftw_malloc, [], [AC_MSG_ERROR([please install libfftw3-dev on your system (sudo apt install libfftw"-dev)])])

# Checks for header files:
AC_CHECK_HEADERS([stdint.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T
AC_TYPE_UINT8_T

# Checks for library functions:
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_CHECK_FUNCS([clock_gettime floor memset pow sqrt strchr strstr])

#################################################################
# export variables
#################################################################
AC_SUBST(PLATFORM)
AC_SUBST(MAKELIB)
AC_SUBST(CC)
AC_SUBST(CXX)
AC_SUBST(CFLAGS)
AC_SUBST(CXXFLAGS)
AC_SUBST(my_CPPFLAGS)
AC_SUBST(LIBS)
AC_SUBST(libtool)
AC_SUBST(MEX_LDEXTENSION)
AC_SUBST(MEX_DIRNAME)
AC_ARG_VAR(PYTHON, [python interpreter])  # set variable as precious
AC_ARG_VAR(MATLAB, [matlab installation (e.g.: MATLAB=/Applications/MATLAB_R2023a.app/)]) # set variable as precious
AC_SUBST(TAR)
AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/ann_1.1.2/src/Makefile
                 src/matlab/Makefile])
AC_OUTPUT
