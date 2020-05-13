# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

from setuptools import setup, find_packages

setup(
    name='pyincore-viz',
    version='0.2.1',
    packages=find_packages(where=".", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={
        '': ['*.ini']
    },
    description='IN-CORE visualization python package',
    long_description=("This package is designed for visualizing the output of pyIncore analyses."
                      "pyIncore is a Python package to analyze and visualize various hazard "
                      "(earthquake, tornado, hurricane etc.) scenarios developed "
                      "by the Center for Risk-Based Community Resilience Planning team from NCSA. "
                      "The development is part of NIST sponsored IN-CORE (Interdependent Networked Community "
                      "Resilience Modeling Environment) initiative. "
                      "pyIncore allows users to apply hazards on infrastructure in selected areas. "
                      "Python framework acceses underlying data through local or remote services "
                      "and facilitates moving and synthesizing results."),
    # TODO need to figure out what are the dependency requirements
    install_requires=[
        "branca==0.3.1",
        "ipyleaflet>=0.12.2",
        "ipywidgets==7.5.0",
        "pandas>=0.24.1",
        "geopandas>=0.6.1",
        "rasterio>=1.1.3",
        "descartes>=1.1.0",
        "contexily>=1.0.0",
        "numpy>=1.16.1",
        "scipy>=1.2.0",
        "networkx>=2.2",
        "owslib>=0.17.1",
        "matplotlib>=2.1.0",
        "plotly>=3.6.0",
        "pytest>=3.9.0",
        "pyincore>=0.6.2"
    ],
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
    ],
    keywords=[
        "infrastructure",
        "resilience",
        "hazards",
        "data discovery",
        "IN-CORE",
        "earthquake",
        "tsunami",
        "tornado",
        "hurricane",
        "dislocation"
    ],
    license="Mozilla Public License v2.0",
    url="https://git.ncsa.illinois.edu/incore/pyincore-viz"
)