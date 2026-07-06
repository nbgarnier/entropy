"""
Configuration and information functions for the entropy module.
"""

import numpy as np
from typing import List, Optional
import os

# Default parameters
k_default = 5

# Sampling defaults
_samp_default = {
    'type': 4,       # adapted
    'Theiler': -4,   # automatic
    'N_eff': 4096,
    'N_real': 10,
}

# Last sampling parameters used
_last_samp = {
    'type': 4,
    'Theiler': 0,
    'Theiler_max': 0,
    'N_eff': 0,
    'N_eff_max': 0,
    'N_real': 0,
    'N_real_max': 0,
}

# Last computation info
_last_info = {
    'std': 0.0,
    'std2': 0.0,
    'n_errors': 0,
    'n_eff_local': 0,
    'n_eff': 0,
    'n_real': 0,
    'Theiler_x': 0,
    'Theiler_y': 0,
}

# Extra info
_extra_info = {
    'data_std': 0.0,
    'data_std_std': 0.0,
}

# 2D Theiler settings
_samp_2d = {
    'type': 2,
    'last_Theiler_y': 0,
}

# Algorithm choice
_mi_algo = 1  # 1 or 2 for Kraskov algorithms
_counting_algo = 1  # 1 for legacy, 2 for ANN

# Multithreading settings
_n_threads = -1  # -1 for auto
_use_threads = True


def get_last_info(verbosity: int = 0) -> List:
    """
    Returns information from the last computation.

    Parameters
    ----------
    verbosity : int
        If > 0, print information to console

    Returns
    -------
    List
        [std, std2, n_errors, n_eff_local, n_eff, n_real, Theiler_x, Theiler_y]
    """
    if verbosity > 0:
        print("from last function call:")
        print(f"- standard deviation(s):      {_last_info['std']:.6f}, {_last_info['std2']:.6f}")
        print(f"- nb of realizations:         {_last_info['n_real']}")
        print(f"- nb of errors encountered:   {_last_info['n_errors']} (total)")
        print(f"- effective nb of points:     {_last_info['n_eff_local']} (per realization)")
        print(f"                              {_last_info['n_eff']} (total)")
        print(f"- Theiler scale               {_last_info['Theiler_x']} (and {_last_info['Theiler_y']} if using a second direction)")

    return [
        _last_info['std'],
        _last_info['std2'],
        _last_info['n_errors'],
        _last_info['n_eff_local'],
        _last_info['n_eff'],
        _last_info['n_real'],
        _last_info['Theiler_x'],
        _last_info['Theiler_y'],
    ]


def get_extra_info(verbosity: int = 0) -> List:
    """
    Returns extra information from the last computation.

    Parameters
    ----------
    verbosity : int
        If > 0, print information to console

    Returns
    -------
    List
        [data_std, data_std_std]
    """
    if verbosity > 0:
        print("from last function call:")
        print(f"- standard deviation of processed input data  : {_extra_info['data_std']:.6f}")
        print(f"- std of this very std                        : {_extra_info['data_std_std']:.6f}")

    return [_extra_info['data_std'], _extra_info['data_std_std']]


def choose_algorithm(algo: int = 1, version: int = 1) -> None:
    """
    Select the algorithm for computing mutual information.

    Parameters
    ----------
    algo : int
        Kraskov algorithm: 1, 2, or 3 (both)
    version : int
        Counting algorithm: 1 (legacy), 2 (ANN-based)
    """
    global _mi_algo, _counting_algo
    _mi_algo = algo
    _counting_algo = version


def set_verbosity(level: int = 1) -> None:
    """
    Set the verbosity level of the library.

    Parameters
    ----------
    level : int
        Verbosity level (0=errors only, 1=warnings, 2+=more detail)
    """
    # In pure Python, we use standard Python logging/warnings
    pass


def get_verbosity() -> None:
    """Get the current verbosity level."""
    print("verbosity level: 1 (standard)")


def set_Theiler(Theiler: int = 4) -> None:
    """
    Set the default Theiler prescription.

    Parameters
    ----------
    Theiler : int or str
        Theiler prescription:
        - 1 or "legacy": tau_Theiler=tau, uniform sampling
        - 2 or "smart": tau_Theiler=max>=tau, uniform sampling
        - 3 or "random": tau_Theiler=tau, random sampling
        - 4 or "adapted": tau_Theiler adaptive
    """
    global _samp_default

    if Theiler in (1, 'legacy'):
        _samp_default['type'] = 1
        _samp_default['N_eff'] = -1
        _samp_default['N_real'] = -1
    elif Theiler in (2, 'smart'):
        _samp_default['type'] = 2
    elif Theiler in (3, 'random'):
        _samp_default['type'] = 3
    elif Theiler in (4, 'adapted'):
        _samp_default['type'] = 4
    else:
        print("please provide a valid prescription (int or string) (see help)")
        return

    _samp_default['Theiler'] = -_samp_default['type']
    print(f"now using Theiler prescription {_samp_default['type']}")


def set_sampling(Theiler: int = 4, N_eff: int = 4096, N_real: int = 10) -> None:
    """
    Set default sampling parameters.

    Parameters
    ----------
    Theiler : int
        Theiler prescription (see set_Theiler)
    N_eff : int
        Number of effective points per realization
    N_real : int
        Number of realizations
    """
    set_Theiler(Theiler)
    if _samp_default['type'] > 0:
        if N_eff > 1:
            _samp_default['N_eff'] = N_eff
        if N_real > 0:
            _samp_default['N_real'] = N_real


def get_sampling(verbosity: int = 1) -> List:
    """
    Get the default sampling parameters.

    Parameters
    ----------
    verbosity : int
        If > 0, print to console

    Returns
    -------
    List
        [type, Theiler, N_eff, N_real]
    """
    if verbosity > 0:
        type_names = {1: "(legacy)", 2: "(smart)", 3: "(random)", 4: "(adapted)"}
        print(f"Theiler prescription : {_samp_default['type']}{type_names.get(_samp_default['type'], '')}")
        print(f"Theiler scale        : {_samp_default['Theiler']}")
        print(f"N_eff                : {_samp_default['N_eff']}")
        print(f"N_realizations       : {_samp_default['N_real']}")

    return [
        _samp_default['type'],
        _samp_default['Theiler'],
        _samp_default['N_eff'],
        _samp_default['N_real'],
    ]


def get_last_sampling(verbosity: int = 0) -> List:
    """
    Get the sampling parameters from the last computation.

    Parameters
    ----------
    verbosity : int
        If > 0, print to console

    Returns
    -------
    List
        [type, Theiler, Theiler_max, N_eff, N_eff_max, N_real, N_real_max]
    """
    if verbosity > 0:
        print("from last function call:")
        print(f"- Theiler prescription : {_last_samp['type']}")
        print(f"- Theiler value        : {_last_samp['Theiler']} max : {_last_samp['Theiler_max']}")
        print(f"- N_eff                : {_last_samp['N_eff']} max : {_last_samp['N_eff_max']}")
        print(f"- N_realizations       : {_last_samp['N_real']} max : {_last_samp['N_real_max']}")

    return [
        _last_samp['type'],
        _last_samp['Theiler'],
        _last_samp['Theiler_max'],
        _last_samp['N_eff'],
        _last_samp['N_eff_max'],
        _last_samp['N_real'],
        _last_samp['N_real_max'],
    ]


def set_Theiler_2d(Theiler: int = 2) -> None:
    """
    Set the 2D Theiler prescription for images.

    Parameters
    ----------
    Theiler : int or str
        - 1 or "minimal": tau_Theiler in each direction
        - 2 or "maximal": max(stride_x, stride_y)
        - 4 or "optimal": sqrt(stride_x^2 + stride_y^2)
    """
    global _samp_2d

    if Theiler in (1, 'minimal'):
        _samp_2d['type'] = 1
    elif Theiler in (2, 'maximal'):
        _samp_2d['type'] = 2
    elif Theiler in (4, 'optimal'):
        _samp_2d['type'] = 4
    else:
        print("please provide a valid prescription (int or string) (see help)")
        return

    print(f"now using 2-d prescription #{_samp_2d['type']}")


def get_Theiler_2d() -> int:
    """Get the current 2D Theiler prescription."""
    return _samp_2d['type']


def multithreading(do_what: str = "info", nb_cores: int = 0) -> None:
    """
    Configure multithreading.

    Parameters
    ----------
    do_what : str
        "info": display current settings
        "auto": use all available cores
        "single": single-threaded
    nb_cores : int
        If > 0, use this many cores
    """
    global _n_threads, _use_threads

    if do_what == "info":
        avail = os.cpu_count() or 1
        current = _n_threads if _n_threads > 0 else avail
        print(f"currently using {current} out of {avail} cores available")
        if _n_threads == -1:
            print(f" (-1 means largest number available, so {avail} here)")
    elif do_what == "auto":
        _use_threads = True
        _n_threads = -1
    elif do_what == "single":
        _use_threads = False
        _n_threads = 1
    elif isinstance(do_what, int) and do_what > 0:
        _use_threads = True
        _n_threads = do_what
    else:
        raise ValueError("invalid parameter value")


def get_threads_number() -> int:
    """Get the current number of threads."""
    if _use_threads:
        if _n_threads == -1:
            return os.cpu_count() or 1
        return _n_threads
    return 1
