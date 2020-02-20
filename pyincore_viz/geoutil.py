# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import os
import json
import folium
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ipyleaflet as ipylft
import ipywidgets as ipywgt
import geopandas as gpd

from pyincore_viz import globals
from branca.colormap import linear
from owslib.wms import WebMapService
from pyincore_viz import PlotUtil
from pyincore import baseanalysis


class GeoUtil:
    """Utility methods for georeferenced data."""

    @staticmethod
    def plot_graph_network(graph, coords):
        """Plot graph.

        Args:
            graph (obj):  A nx graph to be drawn.
            coords (dict): Position coordinates.

        """
        # nx.draw(graph, coords, with_lables=True, font_weithg='bold')

        # other ways to draw
        nx.draw_networkx_nodes(graph, coords, cmap=plt.get_cmap('jet'), node_size=100, node_color='g', with_lables=True, font_weithg='bold')
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
        geom = nx.read_shp(filename)
        node_coords = {k: v for k, v in enumerate(geom.nodes())}
        # create graph
        graph = None
        if is_directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        graph.add_nodes_from(node_coords.keys())
        l = [set(x) for x in geom.edges()]
        edg = [tuple(k for k, v in node_coords.items() if v in sl) for sl in l]

        graph.add_edges_from(edg)

        return graph, node_coords

    """
    creates map window with given inventory with multiple csv file using folder location
    """
    @staticmethod
    def map_csv_from_dir(inventory_dataset, column, file_path=None):
        # converting from fiona to geopandas
        inventory_df = gpd.GeoDataFrame.from_features([feature for feature in inventory_dataset], crs='EPSG3857')
        inventory_df = PlotUtil.remove_null_inventories(inventory_df, 'guid')

        csv_map = GeoUtil.create_basemap_ipylft(inventory_df)

        if file_path is None:
            file_path = os.getcwd()
        data, outfiles = GeoUtil.load_all_data(file_path, column)
        inventory_df = inventory_df.merge(data, on='guid')
        inventory_json = json.loads(inventory_df.to_json())
        GeoUtil.create_map_widgets(outfiles, csv_map, inventory_df, inventory_json)

        return csv_map

    ''' 
    create base ipyleaflet map using geopandas dataframe
    '''
    def create_basemap_ipylft(geo_dataframe):
        ext = geo_dataframe.total_bounds
        cen_x, cen_y = (ext[1] + ext[3]) / 2, (ext[0] + ext[2]) / 2
        m = ipylft.Map(center=(cen_x, cen_y), zoom=12, basemap=ipylft.basemaps.Stamen.Toner, crs='EPSG3857',
                       scroll_wheel_zoom=True)
        return m

    """ 
    loading in all data in output path 
    """
    def load_all_data(path_to_data, column_name):
        temp_outfiles = os.listdir(path_to_data)
        outfiles = []
        for temp_outfile in temp_outfiles:
            file_root, file_extension = os.path.splitext(temp_outfile)
            if file_extension.lower() == '.csv':
                outfiles.append(temp_outfile)
        csv_index = 0
        data = None
        for i, file in enumerate(outfiles):
            filename = os.path.join(path_to_data, file)
            if csv_index == 0:
                data = pd.read_csv(filename, dtype=str)
                # data[column_name] = df.Day.astype(str)
                try:
                    data[file] = data[column_name].astype(float)
                except KeyError as err:
                    print(err, "Error!, Given colum name does not exist or the column is not number.")
                    print("Failed to load the dataset csv file. Process aborted")
                    exit(1)
                # data = data.replace(string_to_num_dict)
                # col_keys = [i for i in data.keys() if 'sample' in i]
                # data[file] = data[col_keys].mean(axis=1)
                data = data[['guid', file]]

            else:
                temp = pd.read_csv(filename, dtype=str)
                temp[file] = temp[column_name].astype(float)

                # temp = temp.replace(string_to_num_dict)		# replacing string values with damage ratio values (e.g. "Moderate" => 0.155)
                # col_keys = [i for i in temp.keys() if 'sample' in i]
                # temp[file] = temp[col_keys].mean(axis=1)
                temp = temp[['guid', file]]
                data = data.merge(temp, on='guid')
            csv_index += 1
        return data, outfiles

    '''
    create csv map's actual widgets
    '''
    def create_map_widgets(outfiles, csv_map, inventory_df, inventory_json):
        csv_dir_map_dropdown = ipywgt.Dropdown(description='Outputfile - 1', options=outfiles, width=500)
        file_control1 = ipylft.WidgetControl(widget=csv_dir_map_dropdown, position='bottomleft')

        # csv_dir_map_dropdown2 = ipywgt.Dropdown(description = 'Outputfile - 2', options = outfiles, width=500)
        # file_control2 = ipylft.WidgetControl(widget=csv_dir_map_dropdown2, position='bottomleft')

        button = ipywgt.Button(description='Generate Map', button_style='info')

        generatemap_control = ipylft.WidgetControl(widget=button, position='bottomleft')

        csv_map.add_control(ipylft.LayersControl(position='topright', style='info'))
        csv_map.add_control(ipylft.FullScreenControl(position='topright'))
        csv_map.add_control(generatemap_control)
        # csv_map.add_control(file_control2)
        csv_map.add_control(file_control1)

        def _on_click(event):
            print('Loading: ', csv_dir_map_dropdown.value)
            key = csv_dir_map_dropdown.value
            _create_choropleth_layer(key)
            print('\n')

        def _create_choropleth_layer(key):
            # vmax_val = max(self.bldg_data_df[key])
            vmax_val = 1
            temp_id = list(range(len(inventory_df['guid'])))
            temp_id = [str(i) for i in temp_id]
            choro_data = dict(zip(temp_id, inventory_df[key]))
            layer = ipylft.Choropleth(geo_data=inventory_json, choro_data=choro_data, colormap=linear.YlOrRd_04,
                                      value_min=0, value_max=vmax_val, border_color='black', style={'fillOpacity': 0.8},
                                      name='CSV map')
            csv_map.add_layer(layer)
            print('done')

        # def _create_legend(self):
        #     legend = linear.YlOrRd_04.scale(0, self.vmax_val)
        #     self.m.colormap = legend
        #     out = ipywgt.Output(layout={'border': '1px solid black'})
        #     with out:
        #         display(legend)
        #     widget_control = ipylft.WidgetControl(widget=out, position='topright')
        #     csv_map.add_control(widget_control)
        #     display(csv_map)

        button.on_click(_on_click)

    @staticmethod
    def get_geopandas_map(geodataframe, width=600, height=400):
        """Get GeoPandas map.

        Args:
            geodataframe (pd.DataFrame): Geo referenced DataFrame.

        Returns:
            pd.DataFrame: A map GeoPandas layers.

        """
        m = folium.Map(width=width, height=height, tiles="Stamen Terrain")
        folium.GeoJson(geodataframe.to_json(), name='hospital').add_to(m)
        ext = geodataframe.total_bounds
        m.fit_bounds([[ext[1], ext[0]], [ext[3], ext[2]]])
        return m

    @staticmethod
    def get_gdf_wms_map(gdf, layers_config,
                        width=600, height=400, url=globals.INCORE_GEOSERVER_WMS_URL):
        """Get GDF (GeoDataFrame) and WMS map.

        Args:
            gdf (GeoDataFrame): A layer with various infrastructure (hospital).
            layers_config (list): Layers configurations with id, name and style.

        Returns:
            obj: A folium map with layers.

        """
        m = folium.Map(width=width, height=height, tiles="Stamen Terrain")
        folium.GeoJson(gdf.to_json(), name='hospital').add_to(m)
        bbox_all = gdf.total_bounds

        for layer in layers_config:
            wms_layer = folium.raster_layers.WmsTileLayer(url, name=layer['name'],
                                                          fmt='image/png',
                                                          transparent=True,
                                                          layers='incore:' + layer['id'],
                                                          styles=layer['style'])

            wms_layer.add_to(m)
            wms = WebMapService(url)
            bbox = wms[layer['id']].boundingBox
            # merge bbox
            if bbox[0] < bbox_all[0]:
                bbox_all[0] = bbox[0]
            if bbox[1] < bbox_all[1]:
                bbox_all[1] = bbox[1]
            if bbox[2] > bbox_all[2]:
                bbox_all[2] = bbox[2]
            if bbox[3] > bbox_all[3]:
                bbox_all[3] = bbox[3]

        m.fit_bounds([[bbox_all[1], bbox_all[0]], [bbox_all[3], bbox_all[2]]])
        return m

    @staticmethod
    def get_wms_map(layers_config: list,
                    width=600, height=400, url=globals.INCORE_GEOSERVER_WMS_URL, ):
        """Get a map with WMS layers.

        Args:
            layers_config (list): Layers configurations with id, name and style.

        Returns:
            obj: A map with WMS layers.

        """
        m = folium.Map(width=width, height=height, tiles="Stamen Terrain")
        bbox_all = [9999, 9999, -9999, -9999]
        for layer in layers_config:
            wms_layer = folium.raster_layers.WmsTileLayer(url, name=layer['name'], fmt='image/png', transparent=True,
                                                          layers='incore:' + layer['id'], styles=layer['style'])
            wms_layer.add_to(m)
            wms = WebMapService(url)
            bbox = wms[layer['id']].boundingBox
            # merge bbox
            if bbox[0] < bbox_all[0]: bbox_all[0] = bbox[0]
            if bbox[1] < bbox_all[1]: bbox_all[1] = bbox[1]
            if bbox[2] > bbox_all[2]: bbox_all[2] = bbox[2]
            if bbox[3] > bbox_all[3]: bbox_all[3] = bbox[3]

        folium.LayerControl().add_to(m)
        bounds = ((bbox_all[1], bbox_all[0]), (bbox_all[3], bbox_all[2]))
        m.fit_bounds(bounds)

        return m

    @staticmethod
    def damage_map_viewer():
        pass

    @staticmethod
    # this method create a map window with single inventory with given column name
    def create_geo_map(inventory_df, key='hazardval'):
        ext = inventory_df.total_bounds
        cen_x, cen_y = (ext[1] + ext[3]) / 2, (ext[0] + ext[2]) / 2
        base_map = ipylft.Map(center=(cen_x, cen_y), zoom=12, basemap=ipylft.basemaps.Stamen.Toner, crs='EPSG3857',
                              scroll_wheel_zoom=True)

        bldg_data_json = json.loads(inventory_df.to_json())
        geo = ipylft.GeoJSON(data=bldg_data_json)
        base_map.add_layer(geo)

        title = key
        guid = 'guid'
        value = 'click icon'
        html = ipywgt.HTML(
            '''
                <h4>{}</h4>
                <p>{}</p>
                <p>{}</p>
            '''.format(title, guid, value))

        widget_control1 = ipylft.WidgetControl(widget=html, position='topright')
        base_map.add_control(widget_control1)

        def on_click(event, feature, **kwargs):
            title = key
            guid = feature['properties']['guid']
            value = feature['properties'][key]
            update_html(title, guid, value)

        def update_html(title, guid, value):
            html.value = '''
                    <h4>{}</h4>
                    <p>{}</p>
                    <p>{}</p>
                '''.format(title, guid, value)

        geo.on_click(on_click)

        return base_map