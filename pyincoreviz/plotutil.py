# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import json

import folium
import geopandas as gpd
import ipyleaflet as ipylft
import ipywidgets as ipywgt
import matplotlib.pyplot as plt
import numpy
from owslib.wms import WebMapService
from pyincoreviz import globals
from scipy.stats import lognorm, norm


class PlotUtil:
    """Plotting utility."""

    @staticmethod
    def sample_lognormal_cdf_alt(mean: float, std: float, sample_size: int):
        """Get values from a lognormal distribution.

        Args:
            mean (float):  A mean of the lognormal distribution.
            std (float):  A standard deviation of the lognormal distribution.
            sample_size (int): Number of samples to generate. Numpy default is 50.

        Returns:
            ndarray, ndarray: X sampling, Y cummulative density values.

        """
        dist = lognorm(s=std, loc=0, scale=numpy.exp(mean))
        start = dist.ppf(0.001)  # cdf inverse
        end = dist.ppf(0.999)  # cdf inverse
        x = numpy.linspace(start, end, sample_size)
        y = dist.cdf(x)
        return x, y

    @staticmethod
    def sample_lognormal_cdf(location: float, scale: float, sample_size: int):
        """Get values from a lognormal distribution.

        Args:
            location (float):  A location parameter.
            scale (float):  A scale parameter.
            sample_size (int): Number of samples to generate. Numpy default is 50.

        Returns:
            ndarray, ndarray: X sampling, Y cummulative density values.

        """
        # convert location and scale parameters to the normal mean and std
        mean = numpy.log(numpy.square(location) / numpy.sqrt(scale + numpy.square(location)))
        std = numpy.sqrt(numpy.log((scale / numpy.square(location)) + 1))
        dist = lognorm(s=std, loc=0, scale=numpy.exp(mean))
        start = dist.ppf(0.001)  # cdf inverse
        end = dist.ppf(0.999)  # cdf inverse
        x = numpy.linspace(start, end, sample_size)
        y = dist.cdf(x)
        return x, y

    @staticmethod
    def sample_normal_cdf(mean: float, std: float, sample_size: int):
        """Get values from a normal distribution.

        Args:
            mean (float):  A mean of the normal distribution.
            std (float):  A standard deviation of the normal distribution.
            sample_size (int): Number of samples to generate. Numpy default is 50.

        Returns:
            ndarray, ndarray: X sampling, Y cummulative density values.

        """
        dist = norm(mean, std)
        start = dist.ppf(0.001)  # cdf inverse
        end = dist.ppf(0.999)  # cdf inverse
        x = numpy.linspace(start, end, sample_size)
        y = dist.cdf(x)
        return x, y

    @staticmethod
    def get_x_y(disttype: str, alpha: float, beta: float):
        """Get arrays of x and y values.

        Args:
            disttype (str):  A distribution type (log normal and normal).
            alpha (float):  A distribution parameter (mostly mean).
            beta (float):  A distribution parameter (mostly standard deviation).

        Returns:
            ndarray, ndarray: X sampling, Y cummulative density values.

        """
        if disttype == 'LogNormal':
            return PlotUtil.sample_lognormal_cdf_alt(alpha, beta, 200)
        if disttype == 'Normal':
            return PlotUtil.sample_lognormal_cdf(alpha, beta, 200)
        if disttype == 'standardNormal':
            return PlotUtil.sample_normal_cdf(alpha, beta, 200)

    @staticmethod
    def get_fragility_plot(fragility_set):
        """Get fragility plot.

        Args:
            fragility_set (obj): A JSON like description of fragility assigned to the
                infrastructure inventory.

        Returns:
            collection: Plot and its style functions.

        """
        for curve in fragility_set['fragilityCurves']:
            x, y = PlotUtil.get_x_y(curve['curveType'], curve['median'], curve['beta'])
            plt.plot(x, y, label=curve['description'])
        plt.xlabel(fragility_set['demandType'] + " (" + fragility_set['demandUnits'] + ")")
        plt.legend()
        return plt

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
    def damage_map_viewer():
        pass

    @staticmethod
    def inventory_to_geodataframe(inventory_dataset):
        fname = inventory_dataset.get_file_path(type="shp")
        inventory_df = gpd.read_file(fname)

        return inventory_df

    @staticmethod
    def remove_null_inventories(inventory_df, key='guid'):
        inventory_df.dropna(subset=[key], inplace=True)

        return inventory_df

    @staticmethod
    def dmg_state2value(damage_result, dmg_ratio_tbl):
        #TODO given damage ratio table, subtitute the damage state with actual mean damage factor
        pass

    @staticmethod
    def merge_inventory_w_dmg(inventory_df, damage_result):
        inventory_df = inventory_df.merge(damage_result, on='guid')

        return inventory_df

    @staticmethod
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


    @staticmethod
    def mean_damage_histogram(mean_damage_dataset, histogram_bins=20, figure_size = (10 , 5), axes_font_size = 12,
                              title_font_size = 12):

        mean_damage = mean_damage_dataset.get_dataframe_from_csv()
        ax = mean_damage['meandamage'].hist(bins=histogram_bins, figsize=figure_size)
        ax.set_title("Mean damage distribution", fontsize=title_font_size)
        ax.set_xlabel("mean damage value", fontsize=axes_font_size)
        ax.set_ylabel("counts", fontsize=axes_font_size)
        fig = ax.get_figure()

        return fig
