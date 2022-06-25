# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

from setuptools import setup, find_packages
import pkg_resources
pkg_resources.extern.packaging.version.Version = pkg_resources.SetuptoolsLegacyVersion

# version number of pyincore
version = '1.5.1'

with open("README.rst", encoding="utf-8") as f:
    readme = f.read()

setup(
    name='pyincore-viz',
    version=version,
    description='IN-CORE visualization python package',
    long_description=readme,
    long_description_content_type='text/x-rst',

    url='https://incore.ncsa.illinois.edu',

    license="Mozilla Public License v2.0",

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

    packages=find_packages(where=".", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={
        '': ['*.ini']
    },

    install_requires=[
        'branca==0.5.0',
        'contextily==1.2.0',
        'deprecated==1.2.13',
        'geopandas==0.11.0',
        'ipyleaflet==0.16.0',
        'ipywidgets==7.7.1',
        'lxml==4.9.0',
        'matplotlib==3.5.2',
        'networkx==2.8.4',
        'numpy==1.23.0',
        'osgeo==0.0.1',
        'owslib==0.26.0',
        'pandas==1.4.3',
        'pillow==9.1.1',
        'pyincore==1.4.2rc5',
        'rasterio==1.2.10'
    ],

    extras_require={
        'test': [
            'pycodestyle==2.8.0',
            'pytest==7.1.2',
            'python-jose==3.3.0',
        ]
    },

    project_urls={
        'Bug Reports': 'https://github.com/IN-CORE/pyincor-vize/issues',
        'Source': 'https://github.com/IN-CORE/pyincor-vize',
    },
)
