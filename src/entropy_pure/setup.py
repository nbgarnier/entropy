"""
setup.py for package "entropy_pure"
Pure Python implementation - no compilation required.
"""
from setuptools import setup, find_packages

setup(
    name='entropy_pure',
    version='4.2.0',
    description="Information Theory tools and entropies for multi-scale analysis (pure Python)",
    author="Nicolas B. Garnier",
    author_email="nicolas.garnier@ens-lyon.fr",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
    ],
    extras_require={
        'dev': ['pytest', 'matplotlib'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
