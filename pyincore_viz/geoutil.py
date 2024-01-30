# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import contextily as ctx
# import geopandas as gpd
# import ipyleaflet as ipylft
# import lxml
# import matplotlib.pyplot as plt
# import networkx as nx
# import rasterio
# import rasterio.plot
# import copy
# import os
# import PIL
# import numpy as np
# import random
# import json
#
# from deprecated.sphinx import deprecated
# from matplotlib import cm
# from pathlib import Path
# from osgeo import gdal
# from osgeo.gdalconst import GA_ReadOnly
# from ipyleaflet import projections
# from owslib.wms import WebMapService
# from pyincore.dataservice import DataService
# from pyincore.hazardservice import HazardService
# from pyincore import Dataset
# from pyincore import NetworkDataset
from pyincore_viz import globals
# from base64 import b64encode
# from io import BytesIO
# from pyincore_viz.plotutil import PlotUtil
# from pyincore_viz.tabledatasetlistmap import TableDatasetListMap as table_list_map
# from pyincore_viz.helpers.common import get_period_and_demand_from_str, get_demands_for_dataset_hazards
# from branca.colormap import linear

# logger = globals.LOGGER


class GeoUtil:
    """Utility methods for Geospatial Visualization"""

    @staticmethod
    def plot_gdf_map(gdf, column, category=False, basemap=True, source=ctx.providers.OpenStreetMap.Mapnik):
        """Plot Geopandas DataFrame.

        Args:
            gdf (obj): Geopandas DataFrame object.
            column (str): A column name to be plot.
            category (bool): Turn on/off category option.
            basemap (bool): Turn on/off base map (e.g. openstreetmap).
            source(obj): source of the Map to be used. examples, ctx.providers.OpenStreetMap.Mapnik (default),
                ctx.providers.Stamen.Terrain, ctx.providers.CartoDB.Positron etc.

        Returns:

        """
        gdf = gdf.to_crs(epsg=3857)
        ax = gdf.plot(figsize=(10, 10), column=column,
                      categorical=category, legend=True)
        if basemap:
            ctx.add_basemap(ax, source=source)
