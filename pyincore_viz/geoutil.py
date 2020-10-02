# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

from pathlib import Path

import contextily as ctx
import geopandas as gpd
import ipyleaflet as ipylft
import ipywidgets as ipywgt
import matplotlib.pyplot as plt
import networkx as nx
import rasterio
import rasterio.plot
import gdal
import copy
import json
import os

from gdalconst import GA_ReadOnly
from pyincore.dataservice import DataService
from pyincore.hazardservice import HazardService
from pyincore import Dataset
from pyincore import NetworkDataset
from pyincore_viz import globals
from owslib.wms import WebMapService
from ipyleaflet import ImageOverlay
from pyincore_viz.plotutil import PlotUtil
from branca.colormap import linear

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
            wms_url (str): URL of WMS server
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
                    print("Error: The layer " + str(dataset.id) + " does not exist in the wms server")
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
            wms_url (str): URL of WMS server

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
    def plot_network_dataset(network_dataset: NetworkDataset, zoom_level=10):
        """Creates map window with Network Dataset visualized

        Args:
            network_dataset (NetworkDataset):  pyincore Network Dataset object
            zoom_level (int): zoom level indicator value for mapping

        Returns:
            m (ipyleaflet.Map): ipyleaflet Map object

        """
        # get node file name path
        link_path = network_dataset.link.file_path
        link_file_name = os.path.basename(link_path)

        # get node file name path
        node_path = network_dataset.node.file_path
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

        cen_lat, cen_lon = (bbox_all[2] + bbox_all[0]) / 2.0, (bbox_all[3] + bbox_all[1]) / 2.0

        # TODO: ipylft doesn't have fit bound methods, we need to find a way to zoom level to show all data
        m = ipylft.Map(center=(cen_lon, cen_lat), zoom=zoom_level, basemap=ipylft.basemaps.Stamen.Toner, crs='EPSG3857',
                       scroll_wheel_zoom=True)
        for entry in geo_data_list:
            m.add_layer(entry)

        m.add_control(ipylft.LayersControl())

        return m

    @staticmethod
    def plot_raster_from_url_or_path(input_path, zoom_level=10):
        """Creates map window with geo-referenced raster file from local or url visualized

            Args:
                input_path (str):  input raster dataset (GeoTiff) file path
                zoom_level (int): zoom level indicator value for mapping

            Returns:
                map (ipyleaflet.Map): ipyleaflet Map object

        """

        boundary = GeoUtil.get_raster_boundary(input_path)
        cen_lat, cen_lon = (boundary[2] + boundary[0]) / 2.0, (boundary[3] + boundary[1]) / 2.0
        map = ipylft.Map(center=(cen_lon, cen_lat), zoom=zoom_level,
                         basemap=ipylft.basemaps.Stamen.Toner, crs='EPSG3857', scroll_wheel_zoom=True)
        image = ImageOverlay(url=input_path, bounds=((boundary[0], boundary[1]), (boundary[2], boundary[3])))
        map.add_layer(image)

        return map

    @staticmethod
    def plot_table_dataset(client, dataset_list=list, column=str, in_source_dataset_id=None):
        """Creates map window with a list of table dataset and source dataset

            Args:
                client (Client): pyincore service Client Object
                dataset_list (list): list of table dataset
                column (str): column name to be plot
                in_source_dataset_id (str): source dataset id, the dafault is None

            Returns:
                GeoUtil.map (ipyleaflet.Map): ipyleaflet Map object

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

        # create base map
        map = GeoUtil.create_basemap_ipylft(inventory_df)

        # merge inventory dataframe and joined table dataframe
        inventory_df = inventory_df.merge(joined_df, on='guid')

        # keep only necessary fields
        keep_list = ['guid', 'geometry']
        for dataset_id in dataset_id_list:
            # dataset_id will be used as a column name to visualize the values in the field
            keep_list.append(dataset_id)
        inventory_df = inventory_df[keep_list]
        inventory_json = json.loads(inventory_df.to_json())

        # set global for button click
        GeoUtil.inventory_json = inventory_json
        GeoUtil.map = map

        GeoUtil.map = GeoUtil.create_map_widgets(dataset_id_list, GeoUtil.map, inventory_df)

        return GeoUtil.map

    @staticmethod
    def merge_table_dataset_with_field(dataset_list: list, column=str, in_source_dataset_id=None):
        """Creates pandas dataframe with all dataset in the list joined with guid and column

        Args:
            dataset_list (list): list of table dataset
            column (str): column name to be plot
            source_dataset (str): source dataset id, default is None

        Returns:
            join_df (dataframe): pandas dataframe with all dataset joined together with guid
            dataset_id_list (list): list of dataset id
            dataset_title_list (list): list of dataset title
            common_source_dataset_id (str): common source dataset id from datasets

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

    @staticmethod
    def create_basemap_ipylft(geo_dataframe):
        """Creates map window with given inventory with multiple table dataset file using folder location

        Args:
            geo_dataframe (DataFrame): Geopandas DataFrame object

        Returns:
            m(ipyleaflet.Map): ipyleaflet Map object

        """
        ext = geo_dataframe.total_bounds
        cen_x, cen_y = (ext[1] + ext[3]) / 2, (ext[0] + ext[2]) / 2
        map = ipylft.Map(center=(cen_x, cen_y), zoom=12, basemap=ipylft.basemaps.Stamen.Toner, scroll_wheel_zoom=True)

        return map

    @staticmethod
    def create_map_widgets(title_list, map, inventory_df):
        """Create and add map widgets into map

        Args:
            title_list (list): list of the file names in the folder

        """
        map_dropdown = ipywgt.Dropdown(description='Outputfile - 1', options=title_list, width=500)
        file_control1 = ipylft.WidgetControl(widget=map_dropdown, position='bottomleft')

        # use the following line when it needs to have another dropdown
        # dropdown2 = ipywgt.Dropdown(description = 'Outputfile - 2', options = title_list2, width=500)
        # file_control2 = ipylft.WidgetControl(widget=dropdown2, position='bottomleft')

        button = ipywgt.Button(description='Generate Map', button_style='info')
        button.on_click(GeoUtil.on_button_clicked)
        map_control = ipylft.WidgetControl(widget=button, position='bottomleft')

        map.add_control(ipylft.LayersControl(position='topright', style='info'))
        map.add_control(ipylft.FullScreenControl(position='topright'))
        map.add_control(map_control)
        # map.add_control(file_control2)      # use the line when it needs to have extra dropdown
        map.add_control(file_control1)

        # set global for button click
        GeoUtil.map_dropdown = map_dropdown
        GeoUtil.inventory_df = inventory_df

        return map

    @staticmethod
    def get_raster_boundary(input_path):
        """Creates boundary list from raster dataset file

            Args:
                input_path (str):  input raster dataset (GeoTiff) file path

            Returns:
                boundary (list): list of boundary values

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
    def on_button_clicked(b):
        """button click action for map

        Args:
            b (action): button click action for tablemap

        """
        print('Loading: ', GeoUtil.map_dropdown.value)
        key = GeoUtil.map_dropdown.value
        GeoUtil.create_choropleth_layer(key)
        print('\n')

    @staticmethod
    def create_choropleth_layer(key):
        """add choropleth layer to map

        Args:
            key (str): selected value from tablemap's layer selection drop down menu

        """

        # vmax_val = max(self.bldg_data_df[key])
        vmax_val = 1
        temp_id = list(range(len(GeoUtil.inventory_df['guid'])))
        temp_id = [str(i) for i in temp_id]
        choro_data = dict(zip(temp_id, GeoUtil.inventory_df[key]))
        layer = ipylft.Choropleth(geo_data=GeoUtil.inventory_json, choro_data=choro_data,
                                  colormap=linear.YlOrRd_04,
                                  value_min=0, value_max=vmax_val, border_color='black', style={'fillOpacity': 0.8},
                                  name='dataset map')
        GeoUtil.map.add_layer(layer)

        print('Done loading layer.')

    # TODO the following method for adding layer should be added in the future
    # def create_legend(self):
    #     legend = linear.YlOrRd_04.scale(0, self.vmax_val)
    #     TableDatasetMapUtil.tablemap.colormap = legend
    #     out = ipywgt.Output(layout={'border': '1px solid black'})
    #     with out:
    #         display(legend)
    #     widget_control = ipylft.WidgetControl(widget=out, position='topright')
    #     TableDatasetMapUtil.tablemap.add_control(widget_control)
    #     display(TableDatasetMapUtil.tablemap)
