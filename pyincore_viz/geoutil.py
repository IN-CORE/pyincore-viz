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
from pyincore_viz import globals as pyincore_viz_globals
from base64 import b64encode
from io import BytesIO
from pyincore_viz.plotutil import PlotUtil
from pyincore_viz.tabledatasetlistmap import TableDatasetListMap as table_list_map
from pyincore_viz.helpers.common import get_period_and_demand_from_str, get_demands_for_dataset_hazards
from branca.colormap import linear

logger = pyincore_viz_globals.LOGGER


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

        """
        gdf = gdf.to_crs(epsg=3857)
        ax = gdf.plot(figsize=(10, 10), column=column,
                      categorical=category, legend=True)
        if basemap:
            ctx.add_basemap(ax, source=source)

    @staticmethod
    def overlay_gdf_with_raster_hazard(gdf, column, raster, category=False, basemap=True,
                                       source=ctx.providers.OpenStreetMap.Mapnik):
        """Overlay Geopandas DataFrame with raster dataset such as earthquake or flood.

        Args:
            gdf (obj): Geopandas DataFrame object.
            column (obj): A column name of gdf to be plot.
            raster (str): A raster hazard dataset id to overlay, such as tif or png dataset
            category (bool): Turn on/off category option.
            basemap (bool): Turn on/off base map (e.g. openstreetmap).
            source(obj): source of the Map to be used. examples, ctx.providers.OpenStreetMap.Mapnik (default),
                ctx.providers.Stamen.Terrain, ctx.providers.CartoDB.Positron etc.

        """
        file_path = Path(raster.local_file_path).joinpath(raster.metadata['fileDescriptors'][0]['filename'])

        # check if the extension is either tif or png
        filename, file_extension = os.path.splitext(file_path)
        if file_extension.lower() != '.png' \
                and file_extension.lower() != '.tiff' and\
                file_extension.lower() != '.tif':
            exit("Error! Given data set is not tif or png. Please check the dataset")

        with rasterio.open(file_path) as r:
            eq_crs = r.crs

        ax = gdf.plot(figsize=(10, 10), column=column, categorical=False, legend=True)
        # TODO there is a problem in crs in following lines.
        #  It should be better to add crs to ctx.add_basemap like following
        #  ctx.add_basemap(ax, crs=eq_crs, source=ctx.providers.OpenStreetMap.Mapnik)
        #  However there is an error in rasterio so the crs part has been omitted.
        #  Since incore only use WGS84 for whole dataset, this should be safe
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ctx.add_basemap(ax, source=file_path, alpha=0.5)

    @staticmethod
    def join_datasets(geodataset, dataset):
        """Join Geospatial Dataset and non-geospatial Dataset.

        Args:
            geodataset (obj): pyincore dataset with geospatial data.
            dataset (obj): pyincore dataset without geospatial data.

        Returns:
            GeoDataFrame: Geopandas DataFrame object.

        """

        # TODO: maybe this method should be moved to Dataset Class
        gdf = gpd.read_file(geodataset.local_file_path)
        df = dataset.get_dataframe_from_csv()
        join_gdf = gdf.set_index("guid").join(df.set_index("guid"))

        return join_gdf

    @staticmethod
    def plot_map(dataset, column, category=False, basemap=True, source=ctx.providers.OpenStreetMap.Mapnik):
        """Plot a map of geospatial dataset.

        Args:
            dataset (obj):  pyincore Dataset object with geospatial data.
            column (str): column name to be plot.
            category (bool): turn on/off category option.
            basemap (bool): turn on/off base map (e.g. openstreetmap).
            source(obj): source of the Map to be used. examples, ctx.providers.OpenStreetMap.Mapnik (default),
                ctx.providers.Stamen.Terrain, ctx.providers.OpenTopoMap.attribution, ctx.providers.CartoDB.Positron etc.

        """
        # maybe this part should be moved to Dataset Class (reading csv to create gdf)
        gdf = gpd.read_file(dataset.local_file_path)
        GeoUtil.plot_gdf_map(gdf, column, category, basemap, source)

    @staticmethod
    def plot_join_map(geodataset, dataset, column, category=False, basemap=True,
                      source=ctx.providers.OpenStreetMap.Mapnik):
        """Plot a map from geospatial dataset and non-geospatial dataset.

        Args:
            geodataset (obj): pyincore Dataset object with geospatial data.
            dataset (obj): pyincore Dataset object without geospatial data.
            column (str): Column name to be plotted.
            category (bool): turn on/off category option.
            basemap (bool): turn on/off base map (e.g. openstreetmap).
            source(obj): source of the Map to be used. examples, ctx.providers.OpenStreetMap.Mapnik (default),
                ctx.providers.Stamen.Terrain, ctx.providers.CartoDB.Positron etc.

        """
        gdf = GeoUtil.join_datasets(geodataset, dataset)
        GeoUtil.plot_gdf_map(gdf, column, category, basemap, source)

    @staticmethod
    def plot_tornado(tornado_id, client, category=False, basemap=True, source=ctx.providers.OpenStreetMap.Mapnik):
        """Plot a tornado path.

        Args:
            tornado_id (str):  ID of tornado hazard.
            client (obj): pyincore service Client Object.
            category (bool): turn on/off category option.
            basemap (bool): turn on/off base map (e.g. openstreetmap).
            source(obj): source of the Map to be used. examples, ctx.providers.OpenStreetMap.Mapnik (default),
                ctx.providers.Stamen.Terrain, ctx.providers.CartoDB.Positron etc.

        """
        # it needs descartes package for polygon plotting
        # getting tornado dataset should be part of Tornado Hazard code
        tornado_dataset_id = HazardService(
            client).get_tornado_hazard_metadata(tornado_id)["hazardDatasets"][0].get('datasetId')
        tornado_dataset = Dataset.from_data_service(
            tornado_dataset_id, DataService(client))
        tornado_gdf = gpd.read_file(tornado_dataset.local_file_path)

        GeoUtil.plot_gdf_map(tornado_gdf, 'ef_rating', category, basemap, source)

    @staticmethod
    def plot_earthquake(earthquake_id, client, demand=None):
        """Plot earthquake raster data.

        Args:
            earthquake_id (str): ID of tornado hazard.
            client (obj): pyincore service Client Object.
            demand (str): A demand type, only applicable to dataset based earthquakes that can have one raster for
                each demand. e.g. PGA, PGV, 0.2 sec SA.

        """
        eq_metadata = HazardService(
            client).get_earthquake_hazard_metadata(earthquake_id)

        eq_dataset_id = None

        if eq_metadata['eqType'] == 'model':
            eq_dataset_id = eq_metadata['hazardDatasets'][0].get('datasetId')
            demand_type = eq_metadata['hazardDatasets'][0].get('demandType')
            period = eq_metadata['hazardDatasets'][0].get('period', "NA")
        else:
            if demand is None:  # get first dataset
                if len(eq_metadata['hazardDatasets']) > 0 and eq_metadata['hazardDatasets'][0]['datasetId']:
                    eq_dataset_id = eq_metadata['hazardDatasets'][0]['datasetId']
                    demand_type = eq_metadata['hazardDatasets'][0]['demandType']
                    period = eq_metadata['hazardDatasets'][0]['period']
                else:
                    raise Exception("No datasets found for the hazard")
            else:  # match the passed demand with a dataset
                demand_parts = get_period_and_demand_from_str(demand)
                demand_type = demand_parts['demandType']
                period = demand_parts['period']

                for dataset in eq_metadata['hazardDatasets']:
                    if dataset['demandType'].lower() == demand_type.lower() and dataset['period'] == period:
                        eq_dataset_id = dataset['datasetId']

                if eq_dataset_id is None:
                    available_demands = get_demands_for_dataset_hazards(eq_metadata['hazardDatasets'])
                    raise Exception("Please provide a valid demand for the earthquake. "
                                    "Available demands for the earthquake are: " + "\n" + "\n".join(available_demands))

        if period > 0:
            title = "Demand Type: " + demand_type.upper() + ", Period: " + str(period)
        else:
            title = "Demand Type: " + demand_type.upper()

        eq_dataset = Dataset.from_data_service(
            eq_dataset_id, DataService(client))
        raster_file_path = Path(eq_dataset.local_file_path).joinpath(
            eq_dataset.metadata['fileDescriptors'][0]['filename'])

        GeoUtil.plot_raster_file_with_legend(raster_file_path, title)

    @staticmethod
    def plot_raster_dataset(dataset_id, client):
        """Plot raster data.

        Args:
            dataset_id (str): ID of tornado hazard.
            client (obj): pyincore service Client Object.

        """
        metadata = DataService(client).get_dataset_metadata(dataset_id)
        # metadata = DataService(client)
        title = metadata['title']

        dataset = Dataset.from_data_service(dataset_id, DataService(client))
        raster_file_path = Path(dataset.local_file_path).\
            joinpath(dataset.metadata['fileDescriptors'][0]['filename'])

        GeoUtil.plot_raster_file_with_legend(raster_file_path, title)

    @staticmethod
    def plot_raster_file_with_legend(file_path, title=None):
        """Plot raster file using matplotlib.

        Args:
            file_path (str): A file path for the raster data.
            title (str): A title for the plot.

        """
        with rasterio.open(file_path) as earthquake_src:
            earthquake_nd = earthquake_src.read(1)

        min = earthquake_nd.min()
        max = earthquake_nd.max()

        # Define the default viridis colormap for viz
        viz_cmap = cm.get_cmap('viridis', 256)

        earthquake_nd = np.flip(earthquake_nd, axis=0)

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        psm = ax.pcolormesh(earthquake_nd, cmap=viz_cmap, rasterized=True, vmin=min, vmax=max)
        fig.colorbar(psm, ax=ax)
        # since the x,y values in the images shows the cell location,
        # this could be misleading. It could be better not showing the x and y value
        plt.axis('off')
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_graph_network(graph, coords):
        """Plot graph.

        Args:
            graph (obj): A nx graph to be drawn.
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
            is_directed (bool): Graph type. True for directed Graph,
                False default for Graph.

        Returns:
            obj: Graph.
            dict: Node coordinates.

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
        """Merge bbox to create bigger bbox to contain both bbox.

        Args:
            bbox1 (list): [min_lat, min_lon, max_lat, max_lon].
            bbox2 (list): [min_lat, min_lon, max_lat, max_lon].

        Returns:
            list: merged bbox [min_lat, min_lon, max_lat, max_lon].

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
    def get_gdf_map(datasets: list):
        """Get ipyleaflet map with list of datasets with geopandas.

        Args:
            datasets (list): a list of pyincore Dataset objects.

        Returns:
            obj: An ipyleaflet Map.

        """

        # TODO: how to add a style for each dataset
        # TODO: performance issue. If there are a lot of data, the browser will crash
        geo_data_list = []
        # (min_lat, min_lon, max_lat, max_lon)
        bbox_all = [9999, 9999, -9999, -9999]

        for i, dataset in enumerate(datasets):
            if isinstance(dataset, Dataset):
                gdf = dataset.get_dataframe_from_shapefile()
                geo_data = ipylft.GeoData(
                    geo_dataframe=gdf, name=dataset.metadata['title'])
            else:
                gdf = dataset
                geo_data = ipylft.GeoData(
                    geo_dataframe=gdf, name="GeoDataFrame_" + str(i))
            geo_data_list.append(geo_data)

            bbox = gdf.total_bounds
            bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)

        m = GeoUtil.get_ipyleaflet_map(bbox_all)

        for entry in geo_data_list:
            m.add_layer(entry)

        m.add_control(ipylft.LayersControl())
        return m

    @staticmethod
    def get_wms_map(datasets: list, wms_url=pyincore_viz_globals.INCORE_GEOSERVER_WMS_URL, layer_check=False):
        """Get a map with WMS layers from list of datasets.

        Args:
            datasets (list): list of pyincore Dataset objects.
            wms_url (str): URL of WMS server.
            layer_check (bool): boolean for checking the layer availability in wms server.

        Returns:
            obj: An ipyleaflet Map.

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
            try:
                wms = WebMapService(wms_url + "?", version='1.1.1')
            except lxml.etree.XMLSyntaxError:
                # The error is caused because it failed to parse the geoserver's return xml.
                # This error will happen in geoserver when there is not complete dataset ingested,
                # and it is very hard to avoid due to current operation setting.
                # It should be passed because this is a proof of the geoserver service is working,
                # and the further layer_check related operation should be stopped
                layer_check = False
            except Exception:
                raise Exception("Geoserver failed to set WMS service.")

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
                    print("Error: The layer " + str(dataset.id) + " does not exist in the wms server")
            wms_layer = ipylft.WMSLayer(url=wms_url, layers=wms_layer_name,
                                        format='image/png', transparent=True, name=dataset.metadata['title'])
            wms_layers.append(wms_layer)

            bbox = dataset.metadata['boundingBox']
            bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)

        m = GeoUtil.get_ipyleaflet_map(bbox_all)

        for layer in wms_layers:
            m.add_layer(layer)

        return m

    @staticmethod
    def get_gdf_wms_map(datasets, wms_datasets, wms_url=pyincore_viz_globals.INCORE_GEOSERVER_WMS_URL):
        """Get a map with WMS layers from list of datasets for geopandas and list of datasets for WMS.

        Args:
            datasets (list): A list of pyincore dataset objects.
            wms_datasets (list): A list of pyincore dataset objects for wms layers.
            wms_url (str): URL of WMS server.

        Returns:
            obj: An ipyleaflet Map.

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

        m = GeoUtil.get_ipyleaflet_map(bbox_all)

        for layer in wms_layers:
            m.add_layer(layer)

        for g in geo_data_list:
            m.add_layer(g)

        m.add_control(ipylft.LayersControl())

        return m

    @staticmethod
    def plot_network_dataset(network_dataset: NetworkDataset):
        """Creates map window with Network Dataset visualized.

        Args:
            network_dataset (obj): pyincore Network Dataset.

        Returns:
            obj: An ipyleaflet Map object, GeoUtil.map (ipyleaflet.Map).

        """
        # get node file name path
        link_path = network_dataset.links.local_file_path
        link_file_name = os.path.basename(link_path)

        # get node file name path
        node_path = network_dataset.nodes.local_file_path
        node_file_name = os.path.basename(node_path)

        # read file using geopandas
        node_gdf = gpd.read_file(node_path)
        link_gdf = gpd.read_file(link_path)

        geo_data_list = []
        # (min_lat, min_lon, max_lat, max_lon)
        bbox_all = [9999, 9999, -9999, -9999]

        # add node data to list
        node_geo_data = ipylft.GeoData(geo_dataframe=node_gdf, name=node_file_name)
        geo_data_list.append(node_geo_data)

        # add link data to list
        link_geo_data = ipylft.GeoData(geo_dataframe=link_gdf, name=link_file_name)
        geo_data_list.append(link_geo_data)

        bbox = link_gdf.total_bounds
        bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)

        m = GeoUtil.get_ipyleaflet_map(bbox_all)

        for entry in geo_data_list:
            m.add_layer(entry)

        m.add_control(ipylft.LayersControl())

        return m

    @staticmethod
    def plot_table_dataset(dataset, client, column=str, category=False, basemap=True,
                           source=ctx.providers.OpenStreetMap.Mapnik):
        """ Creates map window with table dataset.

            Args:
                dataset (obj): pyincore dataset.
                client (obj): pyincore service.
                column (str): column name to be plot.
                category (bool): turn on/off category option.
                basemap (bool): turn on/off base map (e.g. openstreetmap).
                source(obj): source of the Map to be used. examples, ctx.providers.OpenStreetMap.Mapnik (default),
                    ctx.providers.Stamen.Terrain, ctx.providers.CartoDB.Positron etc.

        """
        joined_gdf = GeoUtil.join_table_dataset_with_source_dataset(dataset, client)

        if joined_gdf is not None:
            GeoUtil.plot_gdf_map(joined_gdf, column, category, basemap, source)

    @staticmethod
    def join_table_dataset_with_source_dataset(dataset, client):
        """Creates geopandas dataframe by joining table dataset and its source dataset.

            Args:
                dataset (obj): pyincore dataset.
                client (obj): pyincore service client.

            Returns:
                obj: Geopandas geodataframe object.

        """
        is_source_dataset = False
        source_dataset = None

        # check if the given dataset is table dastaset
        if dataset.metadata['format'] != 'table' and dataset.metadata['format'] != 'csv':
            print("The given dataset is not a table dataset")
            return None

        # check if source dataset exists
        try:
            source_dataset = dataset.metadata['sourceDataset']
            is_source_dataset = True
        except Exception:
            print("There is no source dataset for the give table dataset")

        if is_source_dataset:
            # merge dataset and source dataset
            geodataset = Dataset.from_data_service(source_dataset, DataService(client))
            joined_gdf = GeoUtil.join_datasets(geodataset, dataset)
        else:
            return None

        return joined_gdf

    @staticmethod
    def plot_table_dataset_list_from_single_source(client, dataset_list, column, in_source_dataset_id=None):
        """Creates map window with a list of table dataset and source dataset.

            Args:
                client (obj): pyincore service Client Object.
                dataset_list (list): list of table dataset.
                column (str): column name to be plot.
                in_source_dataset_id (str): source dataset id, the default is None.

            Returns:
                obj: An ipyleaflet Map, GeoUtil.map (ipyleaflet.Map).

            """
        source_dataset_id = None
        if in_source_dataset_id is None:
            joined_df, dataset_id_list, source_dataset_id = \
                GeoUtil.merge_table_dataset_with_field(dataset_list, column)
        else:
            joined_df, dataset_id_list, source_dataset_id = \
                GeoUtil.merge_table_dataset_with_field(dataset_list, column, in_source_dataset_id)

        if source_dataset_id is None:
            raise Exception("There is no sourceDataset id.")

        source_dataset = Dataset.from_data_service(source_dataset_id, DataService(client))
        inventory_df = PlotUtil.inventory_to_geodataframe(source_dataset)
        inventory_df = PlotUtil.remove_null_inventories(inventory_df, 'guid')

        # merge inventory dataframe and joined table dataframe
        inventory_df = inventory_df.merge(joined_df, on='guid')

        # keep only necessary fields
        keep_list = ['guid', 'geometry']
        for dataset_id in dataset_id_list:
            # dataset_id will be used as a column name to visualize the values in the field
            keep_list.append(dataset_id)
        inventory_df = inventory_df[keep_list]

        # create base map
        map = table_list_map()
        map.create_basemap_ipylft(inventory_df, dataset_id_list)

        return map.map

    @staticmethod
    def merge_table_dataset_with_field(dataset_list: list, column=str, in_source_dataset_id=None):
        """Creates pandas dataframe with all dataset in the list joined with guid and column.

        Args:
            dataset_list (list): list of table dataset.
            column (str): column name to be plot.
            in_source_dataset_id (str): source dataset id, default is None.

        Returns:
            obj: Pandas dataframe with all dataset joined together with guid.
            list: A list of dataset id.
            list: A list of dataset title.
            str: A common source dataset id from datasets.

        """
        dataset_id_list = []
        dataset_counter = 0
        join_df = None
        common_source_dataset_id = None

        for dataset in dataset_list:
            if in_source_dataset_id is None:
                source_dataset_id = dataset.metadata["sourceDataset"]
                if dataset_counter > 0:
                    if source_dataset_id != common_source_dataset_id:
                        raise Exception("Error, there are multiple sourceDataset ids")
                else:
                    common_source_dataset_id = copy.copy(source_dataset_id)

            dataset_id = dataset.metadata["id"]
            dataset_id_list.append(dataset_id)
            temp_df = dataset.get_dataframe_from_csv()
            temp_df = temp_df[['guid', column]]
            if dataset_counter == 0:
                join_df = copy.copy(temp_df)
            try:
                if dataset_counter == 0:
                    join_df[dataset_id] = join_df[column].astype(float)
                    join_df = join_df[['guid', dataset_id]]
                else:
                    temp_df[dataset_id] = temp_df[column].astype(float)
                    temp_df = temp_df[['guid', dataset_id]]
                    join_df = join_df.join(temp_df.set_index("guid"), on='guid')
            except KeyError as err:
                logger.debug("Skipping " + dataset_id +
                             ", Given column name does not exist or the column is not number.")
            dataset_counter += 1

        if in_source_dataset_id is not None:
            common_source_dataset_id = in_source_dataset_id

        return join_df, dataset_id_list, common_source_dataset_id

    @deprecated(version="1.2.0", reason="use map_raster_overlay_from_file instead")
    def plot_raster_from_path(input_path):
        """Creates map window with geo-referenced raster file from local or url visualized.

            Args:
                input_path (str): An input raster dataset (GeoTiff) file path.

            Returns:
                obj: An ipyleaflet Map, GeoUtil.map (ipyleaflet.Map).

        """
        return GeoUtil.map_raster_overlay_from_file(input_path)

    @staticmethod
    def map_raster_overlay_from_file(input_path):
        """Creates map window with geo-referenced raster file from local or url visualized.

            Args:
                input_path (str): An input raster dataset (GeoTiff) file path.

            Returns:
                obj: ipyleaflet Map object.

        """
        bbox = GeoUtil.get_raster_boundary(input_path)
        image_url = GeoUtil.create_data_img_url_from_geotiff_for_ipyleaflet(input_path)

        map = GeoUtil.get_ipyleaflet_map(bbox)

        image = ipylft.ImageOverlay(
            url=image_url,
            bounds=((bbox[1], bbox[0]), (bbox[3], bbox[2]))
        )
        map.add_layer(image)

        return map

    @staticmethod
    def get_raster_boundary(input_path):
        """Creates boundary list from raster dataset file.

            Args:
                input_path (str): An input raster dataset (GeoTiff) file path.

            Returns:
                list: A list of boundary values.

        """
        data = gdal.Open(input_path, GA_ReadOnly)
        geoTransform = data.GetGeoTransform()
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * data.RasterXSize
        miny = maxy + geoTransform[5] * data.RasterYSize
        boundary = [minx, miny, maxx, maxy]

        return boundary

    @staticmethod
    def create_data_img_url_from_geotiff_for_ipyleaflet(input_path):
        """Creates boundary list from raster dataset file.

            Args:
                input_path (str): An input raster dataset (GeoTiff) file path.

            Returns:
                str: Data for the png data converted from GeoTiff.

        """
        data = gdal.Open(input_path, GA_ReadOnly)
        cols = data.RasterXSize
        rows = data.RasterYSize
        bands = data.RasterCount
        band = data.GetRasterBand(1)
        tiff_array = band.ReadAsArray(0, 0, cols, rows)
        tiff_norm = tiff_array - np.amin(tiff_array)
        tiff_norm = tiff_norm / np.amax(tiff_norm)
        tiff_norm = np.where(np.isfinite(tiff_array), tiff_norm, 0)
        tiff_im = PIL.Image.fromarray(np.uint8(plt.cm.jet(tiff_norm) * 255))  # specify colormap
        tiff_mask = np.where(np.isfinite(tiff_array), 255, 0)
        mask = PIL.Image.fromarray(np.uint8(tiff_mask), mode='L')
        output_img = PIL.Image.new('RGBA', tiff_norm.shape[::-1], color=None)
        output_img.paste(tiff_im, mask=mask)
        # convert image to png
        f = BytesIO()
        output_img.save(f, 'png')
        data = b64encode(f.getvalue())
        data = data.decode('ascii')
        image_url = 'data:image/png;base64,' + data

        return image_url

    @staticmethod
    def plot_maps_dataset_list(dataset_list, client, column='guid', category=False, basemap=True):
        """Create map window using dataset list. Should be okay whether it is shapefile or geotiff.

            Args:
                dataset_list (list): A list of dataset to be mapped.
                column (str): A column name to be plot.
                client (obj): pyincore service Client.
                category (bool): turn on/off category option.
                basemap (bool): turn on/off base map (e.g. openstreetmap).

            Returns:
                obj: An ipyleaflet Map.

        """
        layer_list = []
        bbox_all = [9999, 9999, -9999, -9999]

        for dataset in dataset_list:
            # check if dataset is shapefile or raster
            try:
                if dataset.metadata['format'].lower() == 'shapefile':
                    gdf = gpd.read_file(dataset.local_file_path)
                    geodata = GeoUtil.create_geodata_from_geodataframe(gdf, dataset.metadata['title'])
                    bbox = gdf.total_bounds
                    bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)

                    layer_list.append(geodata)
                elif dataset.metadata['format'].lower() == 'table' or dataset.metadata['format'] == 'csv':
                    # check source dataset
                    gdf = GeoUtil.join_table_dataset_with_source_dataset(dataset, client)
                    if gdf is None:
                        print(dataset.metadata['title'] + "'s  data format" + dataset.metadata['format'] +
                              " is not supported.")
                    else:
                        geodata = GeoUtil.create_geodata_from_geodataframe(gdf, dataset.metadata['title'])
                        bbox = gdf.total_bounds
                        bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)

                        layer_list.append(geodata)
                elif dataset.metadata['format'].lower() == 'raster' \
                        or dataset.metadata['format'].lower() == 'geotif' \
                        or dataset.metadata['format'].lower() == 'geotif':
                    input_path = dataset.get_file_path('tif')
                    bbox = GeoUtil.get_raster_boundary(input_path)
                    bbox_all = GeoUtil.merge_bbox(bbox_all, bbox)
                    image_url = GeoUtil.create_data_img_url_from_geotiff_for_ipyleaflet(input_path)
                    image = ipylft.ImageOverlay(
                        url=image_url,
                        bounds=((bbox[1], bbox[0]), (bbox[3], bbox[2]))
                    )
                    layer_list.append(image)
                else:
                    print(dataset.metadata['title'] + "'s  data format" + dataset.metadata['format'] +
                          " is not supported.")
            except Exception:
                print("There is a problem in dataset format for ' + dataset.metadata['title']  + '.")

        map = GeoUtil.get_ipyleaflet_map(bbox_all)

        for layer in layer_list:
            map.add_layer(layer)

        map.add_control(ipylft.LayersControl())

        return map

    @staticmethod
    def create_geodata_from_geodataframe(gdf, name):
        """Create map window using dataset list. Should be okay whether it is shapefile or geotiff.

            Args:
                gdf (obj): A geopandas geodataframe.
                name (str): A name of the gdf.

            Returns:
                obj: An ipyleaflet GeoData.

        """
        # create random color
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        geodata = ipylft.GeoData(geo_dataframe=gdf,
                                 style={'color': 'black', 'fillColor': color, 'opacity': 0.05,
                                        'weight': 1.9, 'dashArray': '2', 'fillOpacity': 0.6},
                                 hover_style={'fillColor': 'red', 'fillOpacity': 0.2}, name=name)

        return geodata

    @staticmethod
    def convert_bound_to_ipylft_format(bbox):
        """Convert conventional geodata's bounding box to ipyleaflet bounding box format.

            Args:
                bbox (list): Geodata bounding box with [min_lat, min_lon, max_lat, max_lon].

            Returns:
                list: A bounding box coordinates, [[south, east], [north, west]].

        """
        south = bbox[1]
        east = bbox[0]
        north = bbox[3]
        west = bbox[2]

        bounds = [[south, east], [north, west]]

        return bounds

    @staticmethod
    def calc_center_from_bbox(bbox):
        """Calculate center point location from given bounding box.

            Args:
                bbox (list): Geodata bounding box with [min_lat, min_lon, max_lat, max_lon].

            Returns:
                float: A latitude of center location in the bounding box.
                float: A longitude of center location in the bounding box.

        """
        cen_lat, cen_lon = (bbox[2] + bbox[0]) / 2.0, (bbox[3] + bbox[1]) / 2.0

        return cen_lat, cen_lon

    @staticmethod
    def get_ipyleaflet_map_with_center_location(cen_lon, cen_lat, zoom_level):
        """Creates ipyleaflet map object and fit the map using the center point location and zoom level.

            Args:
                cen_lon (float): Longitude of map's center location.
                cen_lat (float): Latitude of map's center location.
                zoom_level (int): An initial zoom level of the map.

            Returns:
                obj: An ipyleaflet map.

        """
        map = ipylft.Map(center=(cen_lon, cen_lat), zoom=zoom_level, basemap=ipylft.basemaps.OpenStreetMap.Mapnik,
                         crs=projections.EPSG3857, scroll_wheel_zoom=True)

        return map

    @staticmethod
    def get_ipyleaflet_map(bbox=None):
        """Creates ipyleaflet map object and fit the map using the bounding box information.

            Args:
                bbox (list): Geodata bounding box.

            Returns:
                obj: An ipyleaflet map.

        """
        map = ipylft.Map(basemap=ipylft.basemaps.OpenStreetMap.Mapnik, zoom=10,
                         crs=projections.EPSG3857, scroll_wheel_zoom=True)

        if bbox is not None:
            # the boundary information should be converted to ipyleaflet code boundary
            bounds = GeoUtil.convert_bound_to_ipylft_format(bbox)
            map.fit_bounds(bounds)
            # get center for different jupyter versions
            center = GeoUtil.calc_center_from_bbox(bbox)
            # need to reverse x and y
            map.center = [center[1], center[0]]

        map.add_control(ipylft.LayersControl(position='topright'))
        map.add_control(ipylft.FullScreenControl(position='topright'))

        return map

    @staticmethod
    def plot_heatmap(dataset, fld_name, radius=10, blur=10, max=1, multiplier=1, name=""):
        """Creates ipyleaflet map object and fit the map using the bounding box information.

            Args:
                dataset (obj): A dataset to be mapped.
                fld_name (str): A column name to be plot in heat map.
                radius (float): Radius of each "point" of the heatmap.
                blur (float): Amount of blur.
                max (float): Maximum point intensity.
                multiplier (float): A multiplication factor for making fld value to more clearly in the map.
                name (str): name that represents the layer.

            Returns:
                obj: An ipyleaflet map.

        """
        gdf = gpd.read_file(dataset.local_file_path)

        map = GeoUtil.plot_heatmap_from_gdf(gdf, fld_name, radius, blur, max, multiplier, name)

        return map

    @staticmethod
    def plot_heatmap_from_gdf(gdf, fld_name, radius=10, blur=10, max=1, multiplier=1, name=""):
        """Creates ipyleaflet map object and fit the map using the bounding box information.

            Args:
                gdf (GeoDataFrame): GeoPandas geodataframe.
                fld_name (str): column name to be plot in heat map.
                radius (float): Radius of each "point" of the heatmap.
                blur (float): Amount of blur.
                max (float): Maximum point intensity.
                multiplier (float): A multiplication factor for making fld value to more clearly in the map.
                name (str): A name that represents the layer.

            Returns:
                obj: An ipyleaflet map.

        """
        # when the geodataframe is processed, not original(converted directly)
        # by some calculation or join,
        # it loses its geometry object and just become simple object.
        # In that case, it doesn't provide the ability of geometry funtions,
        # such as, calculating the bounding box, get x and y coord, calc centroid, and so on.
        # If this happened to the geodataframe
        # It has to be manually processed by string manipulation
        bbox = []

        # check if geometry works
        is_geometry = True
        first_row = gdf.loc[1]
        if isinstance(first_row.geometry, str):
            is_geometry = False

        # check if the fld_name exists
        if fld_name not in gdf.columns:
            raise Exception("The given field name does not exists")

        # check if the fld_name column is number format
        # used try and except because there were two many objects for numbers
        # to check to see if it is number
        row = gdf.loc[0]
        try:
            row[fld_name] + 5
        except TypeError:
            raise Exception("The given field is not number")

        # create locations placeholder for heatmap using x, y value and field value.
        locations = []

        if (is_geometry):
            if gdf.geom_type[0].lower() != "point" and gdf.geom_type[0].lower() != "polygon" \
                    and gdf.geom_type[0].lower() != "linestring":
                raise Exception("Error, the input dataset's geometry is not supported.")

            # convert polygon to point
            if gdf.geom_type[0].lower() == "polygon":
                points = gdf.copy()
                points.geometry = points['geometry'].centroid
                points.crs = gdf.crs
                gdf = points

            # convert line to point
            if gdf.geom_type[0].lower() == "linestring":
                lines = gdf.copy()
                lines.geometry = lines['geometry'].centroid
                lines.crs = gdf.crs
                gdf = lines

            bbox = gdf.total_bounds
            bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]

            for index, row in gdf.iterrows():
                locations.append([row.geometry.y, row.geometry.x, row[fld_name] * multiplier])
        else:
            # create location information for total bounding box
            # set initial min, max values
            # in this case, it only process when the information is point
            # otherwise, it will need to create a method to create centroid
            # from line and polygon strings, not from geometry,
            # that is kind of out of scope for pyincore-viz.
            # However, if it is needed, maybe it should be included
            # in the future release for pyincore.
            first_geometry = ((first_row.geometry).replace('(', '').replace(')', '')).split()
            if first_geometry[0].lower() != 'point':
                raise Exception("The given geometry is not point.")

            minx = float(first_geometry[1])
            maxx = float(first_geometry[1])
            miny = float(first_geometry[2])
            maxy = float(first_geometry[2])

            for index, row in gdf.iterrows():
                geometry = ((row.geometry).replace('(', '').replace(')', '')).split()
                locations.append([geometry[2], geometry[1], row[fld_name] * multiplier])
                if float(geometry[1]) < minx:
                    minx = float(geometry[1])
                if float(geometry[1]) > maxx:
                    maxx = float(geometry[1])
                if float(geometry[2]) < miny:
                    miny = float(geometry[2])
                if float(geometry[2]) > maxy:
                    maxy = float(geometry[2])

            bbox = [minx, miny, maxx, maxy]

        if name == "":
            name = fld_name

        heatmap = GeoUtil.get_ipyleaflet_heatmap(locations=locations, radius=radius, blur=blur, max=max, name=name)

        map = GeoUtil.get_ipyleaflet_map(bbox)
        map.add_layer(heatmap)
        map.add_control(ipylft.LayersControl(position='topright'))

        return map

    @staticmethod
    def get_ipyleaflet_heatmap(locations=None, radius=10, blur=10, max=1, name=""):
        """Creates ipyleaflet map object and fit the map using the bounding box information.

            Args:
                locations (list): A list of center locations with values.
                radius (float): A radius of each "point" of the heatmap.
                blur (float): Amount of blur.
                max (float): A maximum point intensity.
                name (str): A name that represents the layer.

            Returns:
                obj: An ipyleaflet map.

        """
        # create location list using x, y, and fld value
        heatmap = ipylft.Heatmap(locations=locations, radius=radius, blur=blur, name=name)
        heatmap.max = max
        heatmap.gradient = {0.4: 'red', 0.6: 'yellow', 0.7: 'lime', 0.8: 'cyan', 1.0: 'blue'}

        return heatmap

    @staticmethod
    def plot_local_earthquake(eq_dataset):
        """
        Plot local earthquake data on the map
        """
        demand_type = eq_dataset.demand_type
        demand_units = eq_dataset.demand_units
        hazard_type = eq_dataset.hazard_type
        period = eq_dataset.period
        title = "Demand Type: " + demand_type.upper() + ", Demand Units: " + demand_units + ", Period: " + \
                str(period) + ", Hazard Type: " + hazard_type
        raster_file_path = eq_dataset.dataset.local_file_path

        GeoUtil.plot_raster_file_with_legend(raster_file_path, title)

    @staticmethod
    def plot_local_tsunami(tsu_dataset):
        """
        Plot local tsunami data on the map

        args:
            tsu_dataset (obj): pyincore TsunamiDataset object

        returns:
            none
        """
        demand_type = tsu_dataset.demand_type
        demand_units = tsu_dataset.demand_units
        hazard_type = tsu_dataset.hazard_type
        title = "Demand Type: " + demand_type.upper() + ", Demand Units: " + str(demand_units) + \
                ", Hazard Type: " + hazard_type
        raster_file_path = tsu_dataset.dataset.local_file_path

        GeoUtil.plot_raster_file_with_legend(raster_file_path, title)

    @staticmethod
    def plot_local_flood(flood_dataset):
        """
        Plot local tsunami data on the map

        args:
            tsu_dataset (obj): pyincore TsunamiDataset object

        returns:
            none
        """
        demand_type = flood_dataset.demand_type
        demand_units = flood_dataset.demand_units
        hazard_type = flood_dataset.hazard_type
        title = "Demand Type: " + demand_type.upper() + ", Demand Units: " + str(demand_units) + \
                ", Hazard Type: " + hazard_type
        raster_file_path = flood_dataset.dataset.local_file_path

        GeoUtil.plot_raster_file_with_legend(raster_file_path, title)

    @staticmethod
    def plot_local_hurricane(hur_dataset):
        """
        Plot local hurricane data on the map

        args:
            hur_dataset (obj): pyincore HurricaneDataset object

        returns:
            none
        """
        demand_type = hur_dataset.demand_type
        demand_units = hur_dataset.demand_units
        hazard_type = hur_dataset.hazard_type
        title = "Demand Type: " + demand_type.upper() + ", Demand Units: " + str(demand_units) + \
                ", Hazard Type: " + hazard_type
        raster_file_path = hur_dataset.dataset.local_file_path

        GeoUtil.plot_raster_file_with_legend(raster_file_path, title)

    @staticmethod
    def plot_local_tornado(tornado, id_field):
        """
        Plot local tornado data on the map

        args:
            tornado (obj): pyincore TornadoDataset object
            id_field (str): id field name

        """
        GeoUtil.plot_map(tornado, id_field)

    @staticmethod
    def plot_multiple_vector_dataset(dataset_list):
        """Plot multiple vector datasets on the same map.

            Args:
                dataset_list (list): A list of datasets

            Returns:
                obj: An ipyleaflet map.

        """
        geodata_dic_list = []
        title_list = []
        bbox = None
        # check if the dataset is geodataset and convert dataset to geodataframe
        for dataset in dataset_list:
            try:
                tmp_gpd = gpd.read_file(dataset.local_file_path)
                tmp_min_x = tmp_gpd.bounds.minx.mean()
                tmp_min_y = tmp_gpd.bounds.miny.mean()
                tmp_max_x = tmp_gpd.bounds.maxx.mean()
                tmp_max_y = tmp_gpd.bounds.maxy.mean()

                if bbox is None:
                    bbox = [tmp_min_x, tmp_min_y, tmp_max_x, tmp_max_y]
                tmp_bbox = [tmp_min_x, tmp_min_y, tmp_max_x, tmp_max_y]

                if bbox[0] >= tmp_bbox[0]:
                    bbox[0] = tmp_bbox[0]
                if bbox[1] >= tmp_bbox[1]:
                    bbox[1] = tmp_bbox[1]
                if bbox[2] <= tmp_bbox[2]:
                    bbox[2] = tmp_bbox[2]
                if bbox[3] <= tmp_bbox[3]:
                    bbox[3] = tmp_bbox[3]

                # skim geodataframe only for needed fields
                tmp_fld_list = ['geometry']
                tmp_gpd_skimmed = tmp_gpd[tmp_fld_list]
                tmp_geo_data_dic = json.loads(tmp_gpd_skimmed.to_json())
                geodata_dic_list.append(tmp_geo_data_dic)
                title_list.append(dataset.metadata["title"])

            except Exception:
                raise ValueError("Given dataset might not be a geodataset or has an error in the attribute")

        out_map = GeoUtil.get_ipyleaflet_map(bbox)

        for geodata_dic, title in zip(geodata_dic_list, title_list):
            # add data to  map
            tmp_layer = ipylft.GeoJSON(
                data=geodata_dic,
                style={
                    'opacity': 1, 'fillOpacity': 0.8, 'weight': 1
                },
                hover_style={
                    'color': 'white', 'dashArray': '0', 'fillOpacity': 0.5
                },
                style_callback=GeoUtil.random_color,
                name=title
            )

            out_map.add_layer(tmp_layer)

        return out_map

    @staticmethod
    def random_color(feature):
        """Creates random color for ipyleaflet map feature

            Args:
                feature (obj): geodataframe feature

            Returns:
                obj: dictionary for color

        """
        return {
            'color': 'black',
            'fillColor': random.choice(['red', 'yellow', 'purple', 'green', 'orange', 'blue', 'magenta']),
        }

    @staticmethod
    def plot_choropleth_multiple_fields_from_single_dataset(dataset, field_list):
        """Make choropleth map using multiple fields from single dataset.

            Args:
                dataset (list): A dataset to be mapped
                field_list (list): A list of fields in the dataset

            Returns:
                obj: An ipyleaflet map.

        """
        in_gpd = None
        center_x = None
        center_y = None
        bbox = None

        # check if the dataset is geodataset and convert dataset to geodataframe
        try:
            in_gpd = gpd.read_file(dataset.local_file_path)
            center_x = in_gpd.bounds.minx.mean()
            center_y = in_gpd.bounds.miny.mean()
            title = dataset.metadata["title"]
            bbox = in_gpd.total_bounds

        except Exception:
            raise ValueError("Given dataset might not be a geodataset or has an error in the attribute")

        # skim geodataframe only for needed fields
        field_list.append('geometry')
        in_gpd_tmp = in_gpd[field_list]
        geo_data_dic = json.loads(in_gpd_tmp.to_json())

        out_map = GeoUtil.get_ipyleaflet_map(bbox)

        for fld in field_list:
            if fld != 'geometry':
                tmp_choro_data = GeoUtil.create_choro_data_from_pd(in_gpd, fld)
                # add choropleth data to  map
                tmp_layer = ipylft.Choropleth(
                    geo_data=geo_data_dic,
                    choro_data=tmp_choro_data,
                    colormap=linear.YlOrRd_04,
                    border_color='black',
                    style={'fillOpacity': 0.8},
                    name=fld
                )

                out_map.add_layer(tmp_layer)

        return out_map

    @staticmethod
    def plot_choropleth_multiple_dataset(dataset_list, field_list, zoom_level=10):
        """Make choropleth map using multiple dataset.

            Args:
                dataset_list (list): A list of dataset to be mapped
                field_list (list): A list of fields in the dataset.
                        The order of the list should be matched with the order of dataset list
                zoom_level (int): Zoom level

            Returns:
                obj: An ipyleaflet map.

        """
        geodata_dic_list = []
        choro_data_list = []
        title_list = []
        bbox = None

        # check the size of dataset list and field list
        if len(dataset_list) != len(field_list):
            raise Exception("The dataset list size and field list size doesn't match")

        # check if the dataset is geodataset and convert dataset to geodataframe
        for dataset, fld in zip(dataset_list, field_list):
            try:
                tmp_gpd = gpd.read_file(dataset.local_file_path)
                # the title should be unique otherwise ipyleaflet will not understand correctly
                tmp_title = dataset.metadata["title"] + ": " + str(fld)
                tmp_min_x = tmp_gpd.bounds.minx.mean()
                tmp_min_y = tmp_gpd.bounds.miny.mean()
                tmp_max_x = tmp_gpd.bounds.maxx.mean()
                tmp_max_y = tmp_gpd.bounds.maxy.mean()

                if bbox is None:
                    bbox = [tmp_min_x, tmp_min_y, tmp_max_x, tmp_max_y]
                tmp_bbox = [tmp_min_x, tmp_min_y, tmp_max_x, tmp_max_y]

                if bbox[0] >= tmp_bbox[0]:
                    bbox[0] = tmp_bbox[0]
                if bbox[1] >= tmp_bbox[1]:
                    bbox[1] = tmp_bbox[1]
                if bbox[2] <= tmp_bbox[2]:
                    bbox[2] = tmp_bbox[2]
                if bbox[3] <= tmp_bbox[3]:
                    bbox[3] = tmp_bbox[3]

                # skim geodataframe only for needed fields
                tmp_fld_list = [fld, 'geometry']
                tmp_gpd_skimmed = tmp_gpd[tmp_fld_list]
                tmp_geo_data_dic = json.loads(tmp_gpd_skimmed.to_json())
                tmp_choro_data = GeoUtil.create_choro_data_from_pd(tmp_gpd_skimmed, fld)
                geodata_dic_list.append(tmp_geo_data_dic)
                choro_data_list.append(tmp_choro_data)
                title_list.append(tmp_title)

            except Exception:
                raise Exception("Not a geodataset")

        # calculate center point
        center_x = ((bbox[2] - bbox[0]) / 2) + bbox[0]
        center_y = ((bbox[3] - bbox[1]) / 2) + bbox[1]

        out_map = GeoUtil.get_ipyleaflet_map(bbox)

        for geodata_dic, choro_data, title in zip(geodata_dic_list, choro_data_list, title_list):
            # add choropleth data to  map
            tmp_layer = ipylft.Choropleth(
                geo_data=geodata_dic,
                choro_data=choro_data,
                colormap=linear.YlOrRd_04,
                border_color='black',
                style={'fillOpacity': 0.8},
                name=title
            )

            out_map.add_layer(tmp_layer)

        return out_map

    @staticmethod
    def create_choro_data_from_pd(pd, key):
        """Create choropleth's choro-data from dataframe.

        Args:
            pd (object): an Input dataframe.
            key (str): a string for dictionary key
        Returns:
            obj : A dictionary of dataframe

        """
        print("create choropleth data for", key)
        temp_id = list(range(len(pd[key])))
        temp_id = [str(i) for i in temp_id]
        choro_data = dict(zip(temp_id, pd[key]))

        # # when the fld is number
        # # check the minimum value to use it to nan value, since nan value makes an error.
        # min_val = pd[key].min()
        # for item in choro_data:
        #     if isnan(choro_data[item]):
        #         choro_data[item] = 0
        # # when the fld is not number
        # # fill empty value as blank

        return choro_data

    @staticmethod
    def plot_local_hazard(dataset):
        """Plot hazard dataset on the map

        args:
            dataset (obj): pyincore HazardDataset object

        returns:
            none
        """
        hazard_type = dataset.hazard_type

        if hazard_type.lower() == "earthquake":
            if len(dataset.hazardDatasets) > 1:
                for earthquake in dataset.hazardDatasets:
                    GeoUtil.plot_local_earthquake(earthquake)
            else:
                GeoUtil.plot_local_earthquake(dataset.hazardDatasets[0])
        elif hazard_type.lower() == "tsunami":
            if len(dataset.hazardDatasets) > 1:
                for tsunami in dataset.hazardDatasets:
                    GeoUtil.plot_local_tsunami(tsunami)
            else:
                GeoUtil.plot_local_tsunami(dataset.hazardDataset[0])
        elif hazard_type.lower() == "flood":
            if len(dataset.hazardDatasets) > 1:
                for flood in dataset.hazardDatasets:
                    GeoUtil.plot_local_flood(flood)
            else:
                GeoUtil.plot_local_flood(dataset.hazardDatasets[0])
        elif hazard_type.lower() == "hurricane":
            if len(dataset.hazardDatasets) > 1:
                for hurricane in dataset.hazardDatasets:
                    GeoUtil.plot_local_hurricane(hurricane)
            else:
               GeoUtil.plot_local_hurricane(dataset.hazardDatasets[0])
        elif hazard_type.lower() == "tornado":
            id_field = dataset.EF_RATING_FIELD
            if len(dataset.hazardDatasets) > 1:
                for tornado in dataset.hazardDatasets:
                    GeoUtil.plot_local_tornado(tornado.dataset, id_field)
            else:
                GeoUtil.plot_local_tornado(dataset.hazardDatasets[0].dataset, id_field)
