# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import contextily as ctx
import geopandas as gpd
import ipyleaflet as ipylft
import lxml
import matplotlib.pyplot as plt
import networkx as nx
import rasterio
import rasterio.plot
import copy
import os
import PIL
import numpy as np
import random
import json
import contextily as ctx

from deprecated.sphinx import deprecated
from matplotlib import cm
from pathlib import Path
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from ipyleaflet import projections
from owslib.wms import WebMapService
from pyincore.dataservice import DataService
from pyincore.hazardservice import HazardService
from pyincore import Dataset
from pyincore import NetworkDataset
from pyincore_viz import globals
from base64 import b64encode
from io import BytesIO
from pyincore_viz.plotutil import PlotUtil
from pyincore_viz.tabledatasetlistmap import TableDatasetListMap as table_list_map
from pyincore_viz.helpers.common import get_period_and_demand_from_str, get_demands_for_dataset_hazards
from branca.colormap import linear
import importlib


class GeoUtil:
    """Utility methods for Geospatial Visualization"""

    @staticmethod
    def visualize(dataset, **kwargs):
        """Base visualize method that dynamically imports the necessary modules.

            Args:
                dataset (obj): pyincore dataset without geospatial data.

            Returns:
                None

        """
        # data types that needs to use pop_results_table visualization
        pop_result_table_data_types = ['incorehousingunitallocation']

        try:
            module_name = ""
            # split by namespace, capitalize then join
            for item in dataset.data_type.split(":"):
                module_name += item.capitalize()

            # load module
            # e.g. module_name = IncoreHousingUnitAllocation
            # this is a special case for using popresultstable
            if module_name.lower() in pop_result_table_data_types:
                module_name = "PopResultsTable"

            module = importlib.import_module("pyincore_viz.analysis." + module_name.lower())
            print("Loaded pyincore_viz.analysis." + module_name.lower() + " module successfully.")

            # load class
            analysis_class = getattr(module, module_name)

            # run vis
            return analysis_class.visualize(dataset, **kwargs)

        except Exception:
            raise ValueError("Fail to dynamically import dataset to its corresponding class. Please double "
                             "check the data_type of the dataset!")
