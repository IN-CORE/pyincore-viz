# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

from pathlib import Path

import contextily as ctx
import geopandas as gpd
import ipyleaflet as ipylft
import matplotlib.pyplot as plt
import networkx as nx
import rasterio
import rasterio.plot

from pyincore import Dataset
from pyincore.dataservice import DataService
from pyincore.hazardservice import HazardService
from pyincore_viz import globals
from owslib.wms import WebMapService
from pyincore_viz.csvmaputil import CsvMapUtil

logger = globals.LOGGER

class GeoUtil:
    """Utility methods for Geospatial Visualization"""

    @staticmethod
    def plot_gdf_map(gdf, column, category=False, basemap=True):
        """Plot Geopandas DataFrame

        Args:
            gdf (GeoDataFrame):  Geopandas DataFarme object
            column (str): column name to be plot
            category (boolean): turn on/off category option
            basemap (boolean): turn on/off base map (e.g. openstreetmap)

        """
        gdf = gdf.to_crs(epsg=3857)
        ax = gdf.plot(figsize=(10, 10), column=column,
                      categorical=category, legend=True)
        if basemap:
            ctx.add_basemap(ax)

    @staticmethod
    def join_datasets(geodataset, dataset):
        """Join Geospatial Dataset and non-geospatial Dataset

        Args:
            geodataset (Dataset):  pyincore Dataset object with geospatial data
            dataset (Dataset): pyincore Dataset object without geospatial data

        Returns:
            GeoDataFrame: Geopandas DataFrame object

        """

        # TODO: maybe this method should be moved to Dataset Class
        gdf = gpd.read_file(geodataset.local_file_path)
        df = dataset.get_dataframe_from_csv()
        join_gdf = gdf.set_index("guid").join(df.set_index("guid"))

        return join_gdf

    @staticmethod
    def plot_map(dataset, column, category=False, basemap=True):
        """Plot a map of geospatial dataset

        Args:
            dataset (Dataset):  pyincore Dataset object with geospatial data
            column (str): column name to be plot
            category (boolean): turn on/off category option
            basemap (boolean): turn on/off base map (e.g. openstreetmap)

        """
        # maybe this part should be moved to Dataset Class (reading csv to create gdf)
        gdf = gpd.read_file(dataset.local_file_path)
        GeoUtil.plot_gdf_map(gdf, column, category, basemap)

    @staticmethod
    def plot_join_map(geodataset, dataset, column, category=False, basemap=True):
        """Plot a map from geospatial dataset and non-geospatial dataset

        Args:
            geodataset (Dataset):  pyincore Dataset object with geospatial data
            dataset (Dataset): pyincore Dataset object without geospatial data
            column (str): column name to be plot
            category (boolean): turn on/off category option
            basemap (boolean): turn on/off base map (e.g. openstreetmap)
        """
        gdf = GeoUtil.join_datasets(geodataset, dataset)
        GeoUtil.plot_gdf_map(gdf, column, category, basemap)

    @staticmethod
    def plot_tornado(tornado_id, client, category=False, basemap=True):
        """Plot a tornado path

        Args:
            tornado_id (str):  ID of tornado hazard
            client (Client): pyincore service Client Object
            category (boolean): turn on/off category option
            basemap (boolean): turn on/off base map (e.g. openstreetmap)
        """
        # it needs descartes pakcage for polygon plotting
        # getting tornado dataset should be part of Tornado Hazard code
        tornado_dataset_id = HazardService(
            client).get_tornado_hazard_metadata(tornado_id)['datasetId']
        tornado_dataset = Dataset.from_data_service(
            tornado_dataset_id, DataService(client))
        tornado_gdf = gpd.read_file(tornado_dataset.local_file_path)

        GeoUtil.plot_gdf_map(tornado_gdf, 'ef_rating', category, basemap)

    @staticmethod
    def plot_earthquake(earthquake_id, client):
        """Plot earthquake raster data

        Args:
            earthquake_id (str):  ID of tornado hazard
            client (Client): pyincore service Client Object

        """
        eq_metadata = HazardService(
            client).get_earthquake_hazard_metadata(earthquake_id)
        eq_dataset_id = eq_metadata['rasterDataset']['datasetId']

        eq_dataset = Dataset.from_data_service(
            eq_dataset_id, DataService(client))
        raster_file_path = Path(eq_dataset.local_file_path).joinpath(
            eq_dataset.metadata['fileDescriptors'][0]['filename'])
        raster = rasterio.open(raster_file_path)
        rasterio.plot.show(raster)

    @staticmethod
    def plot_graph_network(graph, coords):
        """Plot graph.

        Args:
            graph (obj):  A nx graph to be drawn.
            coords (dict): Position coordinates.

        """
        # TODO: need to use dataset for input arguements
        # nx.draw(graph, coords, with_lables=True, font_weithg='bold')

        # other ways to draw
        nx.draw_networkx_nodes(graph, coords, cmap=plt.get_cmap(
            'jet'), node_size=100, node_color='g', with_lables=True, font_weithg='bold')
        nx.draw_networkx_labels(graph, coords)
        nx.draw_networkx_edges(graph, coords, edge_color='r', arrows=True)
        plt.show()

    @staticmethod
    def get_network_graph(filename, is_directed=False):
        """Get network graph from filename.

        Args:
            filename (str):  A name of a geo dataset resource recognized by Fiona package.
            is_directed (bool, optional (Defaults to False)): Graph type. True for directed Graph,
                False for Graph.

        Returns:
            obj, dict: Graph and node coordinates.

        """

        # TODO: need to use dataset for input arguements

        geom = nx.read_shp(filename)
        node_coords = {k: v for k, v in enumerate(geom.nodes())}
        # create graph
        graph = None
        if is_directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        graph.add_nodes_from(node_coords.keys())
        list = [set(x) for x in geom.edges()]
        edg = [tuple(k for k, v in node_coords.items() if v in sl) for sl in list]

        graph.add_edges_from(edg)

        return graph, node_coords

    @staticmethod
    def merge_bbox(bbox1, bbox2):
        """merge bbox to create bigger bbox to contain both bbox

        Args:
            bbox1 (list): [min_lat, min_lon, max_lat, max_lon]
            bbox2 (list): [min_lat, min_lon, max_lat, max_lon]

        Returns:
            list: merged bbox [min_lat, min_lon, max_lat, max_lon]

        """
        bbox = [bbox1[0], bbox1[1], bbox1[2], bbox1[3]]

        if bbox2[0] < bbox1[0]:
            bbox[0] = bbox2[0]
        if bbox2[1] < bbox1[1]:
            bbox[1] = bbox2[1]
        if bbox2[2] > bbox1[2]:
            bbox[2] = bbox2[2]
        if bbox2[3] > bbox1[3]:
            bbox[3] = bbox2[3]

        return bbox

    @staticmethod
    def get_gdf_map(datasets: list, zoom_level):
        """Get ipyleaflet map with list of datasets with geopandas .

        Args:
            datasets (list): a list of pyincore Dataset objects
            zoom_level (int): initial zoom level

        Returns:
            ipyleaflet.Map: ipyleaflet Map object

        """

        # TODO: how to add a style for each dataset
        # TODO: performance issue. If there are a lot of data, the browser will crash
        geo_data_list = []
        # (min_lat, min_lon, max_lat, max_lon)
        bbox_all = [9999, 9999, -9999, -9999]

        for dataset in datasets:
            # maybe this part should be moved to Dataset Class
            gdf = gpd.read_file(dataset.local_file_path)
            geo_data = ipylft.GeoData(
                geo_dataframe=gdf, name=dataset.metadata['title'])
            geo_data_list.append(geo_data)

            bbox = gdf.total_bounds
            bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)

        cen_lat, cen_lon = (bbox_all[2] + bbox_all[0]) / 2.0, (bbox_all[3] + bbox_all[1]) / 2.0

        # TODO: ipylft doesn't have fit bound methods, we need to find a way to zoom level to show all data
        m = ipylft.Map(center=(cen_lon, cen_lat), zoom=zoom_level, basemap=ipylft.basemaps.Stamen.Toner, crs='EPSG3857',
                       scroll_wheel_zoom=True)
        for entry in geo_data_list:
            m.add_layer(entry)

        m.add_control(ipylft.LayersControl())
        return m

    @staticmethod
    def get_wms_map(datasets: list, zoom_level, wms_url=globals.INCORE_GEOSERVER_WMS_URL, layer_check=True):
        """Get a map with WMS layers from list of datasets

        Args:
            datasets (list): list of pyincore Dataset objects
            zoom_level (int): initial zoom level
            wmr_url (str): URL of WMS server
            layer_check (bool): boolean for checking the layer availability in wms server

        Returns:
            obj: A ipylfealet Map object

        """
        # TODO: how to add a style for each WMS layers (pre-defined styles on WMS server)
        wms_layers = []
        # (min_lat, min_lon, max_lat, max_lon)
        bbox_all = [9999, 9999, -9999, -9999]
        # the reason for checking this layer_check on/off is that
        # the process could take very long time based on the number of layers in geoserver.
        # the process could be relatively faster if there are not many layers in the geoserver
        # but the processing time could increase based upon the increase of the layers in the server
        # by putting on/off for this layer checking, it could make the process faster.
        if layer_check:
            wms = WebMapService(wms_url + "?", version='1.1.1')
        for dataset in datasets:
            wms_layer_name = 'incore:' + dataset.id
            # check availability of the wms layer
            # TODO in here, the question is the, should this error quit whole process
            # or just keep going and show the error message for only the layer with error
            # if it needs to throw an error and exit the process, use following code block
            # if layer_check:
            #     wms[dataset.id].boundingBox
            # else:
            #     raise KeyError(
            #         "Error: The layer " + str(dataset.id) + " does not exist in the wms server")
            # if it needs to keep going with showing all the layers, use following code block
            if layer_check:
                try:
                    wms[dataset.id].boundingBox
                except KeyError:
                    logger.error("Error: The layer " + str(dataset.id) + " does not exist in the wms server")
            wms_layer = ipylft.WMSLayer(url=wms_url, layers=wms_layer_name,
                                        format='image/png', transparent=True, name=dataset.metadata['title'])
            wms_layers.append(wms_layer)

            bbox = dataset.metadata['boundingBox']
            bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)

        cen_lat, cen_lon = (bbox_all[2] + bbox_all[0]) / 2.0, (bbox_all[3] + bbox_all[1]) / 2.0

        # TODO: ipylft doesn't have fit bound methods, we need to find a way to zoom level to show all data
        m = ipylft.Map(center=(cen_lon, cen_lat), zoom=zoom_level,
                       basemap=ipylft.basemaps.Stamen.Toner, crs='EPSG3857', scroll_wheel_zoom=True)
        for layer in wms_layers:
            m.add_layer(layer)

        m.add_control(ipylft.LayersControl())

        return m

    @staticmethod
    def get_gdf_wms_map(datasets, wms_datasets, zoom_level, wms_url=globals.INCORE_GEOSERVER_WMS_URL):
        """Get a map with WMS layers from list of datasets for geopandas and list of datasets for WMS

        Args:
            datasets (list): list of pyincore Dataset objects
            wms_datasets (list): list of pyincore Dataset objects for wms layers
            zoom_level (int): initial zoom level
            wmr_url (str): URL of WMS server

        Returns:
            obj: A ipylfealet Map object

        """

        # TODO: how to add a style for each WMS layers (pre-defined styules on WMS server) and gdf layers

        # (min_lat, min_lon, max_lat, max_lon)
        bbox_all = [9999, 9999, -9999, -9999]

        geo_data_list = []
        for dataset in datasets:
            # maybe this part should be moved to Dataset Class
            gdf = gpd.read_file(dataset.local_file_path)
            geo_data = ipylft.GeoData(
                geo_dataframe=gdf, name=dataset.metadata['title'])
            geo_data_list.append(geo_data)

            bbox = gdf.total_bounds
            bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)

        wms_layers = []
        for dataset in wms_datasets:
            wms_layer_name = 'incore:' + dataset.id
            wms_layer = ipylft.WMSLayer(url=wms_url, layers=wms_layer_name, format='image/png',
                                        transparent=True, name=dataset.metadata['title'] + '-WMS')
            wms_layers.append(wms_layer)

            bbox = dataset.metadata['boundingBox']
            bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)

        cen_lat, cen_lon = (bbox_all[2] + bbox_all[0]) / 2.0, (bbox_all[3] + bbox_all[1]) / 2.0

        # TODO: ipylft doesn't have fit bound methods, we need to find a way to zoom level to show all data
        m = ipylft.Map(center=(cen_lon, cen_lat), zoom=zoom_level,
                       basemap=ipylft.basemaps.Stamen.Toner, crs='EPSG3857', scroll_wheel_zoom=True)
        for layer in wms_layers:
            m.add_layer(layer)

        for g in geo_data_list:
            m.add_layer(g)

        m.add_control(ipylft.LayersControl())

        return m

    @staticmethod
    def map_csv_from_dir(inventory_dataset, column, file_path):
        """Creates map window with given inventory with multiple csv files using folder location

        Args:
            inventory_dataset (Dataset):  pyincore inventory Dataset object
            column (str): column name to use for the mapping visualization
            file_path (str): file path that contains csv files

        Returns:
            csvmap (ipyleaflet.Map): ipyleaflet Map object

        """
        csvmap = CsvMapUtil.generate_map_csv_from_dir(inventory_dataset, column, file_path)

        return csvmap
