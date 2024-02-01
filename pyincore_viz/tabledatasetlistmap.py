# Copyright (c) 2020 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import ipyleaflet as ipylft
import ipywidgets as ipywgt
import json

from pyincore_viz import globals
from branca.colormap import linear

logger = globals.LOGGER


class TableDatasetListMap:
    """Mapping class for visualizing list of Table Dataset"""

    def __init__(self):
        self.map = ipylft.Map(center=(0, 0), zoom=12, basemap=ipylft.basemaps.OpenStreetMap.Mapnik,
                              scroll_wheel_zoom=True)

    def create_basemap_ipylft(self, geo_dataframe, title_list):
        """Creates map window with given inventory with multiple table dataset file using folder location.

        Args:
            geo_dataframe (obj): Geopandas DataFrame.
            title_list (list): A list of the file names in the folder.

        Returns:
            obj: ipyleaflet map.

        """
        ext = geo_dataframe.total_bounds
        cen_x, cen_y = (ext[1] + ext[3]) / 2, (ext[0] + ext[2]) / 2

        # create base ipyleaflet map
        self.map = ipylft.Map(center=(cen_x, cen_y), zoom=12,
                              basemap=ipylft.basemaps.OpenStreetMap.Mapnik, scroll_wheel_zoom=True)

        # add map widgets
        self.map = self.create_map_widgets(title_list, self.map, geo_dataframe)

        return self.map

    def create_map_widgets(self, title_list, map, inventory_df):
        """Create and add map widgets into map.

        Args:
            title_list (list): A list of the file names in the folder.

        Returns:

        """
        map_dropdown = ipywgt.Dropdown(description='Outputfile - 1', options=title_list, width=500)
        file_control1 = ipylft.WidgetControl(widget=map_dropdown, position='bottomleft')

        # use the following line when it needs to have another dropdown
        # dropdown2 = ipywgt.Dropdown(description = 'Outputfile - 2', options = title_list2, width=500)
        # file_control2 = ipylft.WidgetControl(widget=dropdown2, position='bottomleft')

        button = ipywgt.Button(description='Generate Map', button_style='info')
        button.on_click(self.on_button_clicked)
        map_control = ipylft.WidgetControl(widget=button, position='bottomleft')

        map.add_control(ipylft.LayersControl(position='topright', style='info'))
        map.add_control(ipylft.FullScreenControl(position='topright'))
        map.add_control(map_control)
        # map.add_control(file_control2)      # use the line when it needs to have extra dropdown
        map.add_control(file_control1)

        # set global for button click
        self.map_dropdown = map_dropdown
        self.inventory_df = inventory_df
        self.inventory_json = json.loads(inventory_df.to_json())

        return map

    def on_button_clicked(self, b):
        """A button click action for map.

        Args:
            b (action): A button click action for tablemap.

        Returns:

        """
        print('Loading: ', self.map_dropdown.value)
        key = self.map_dropdown.value
        self.create_choropleth_layer(key)
        print('\n')

    def create_choropleth_layer(self, key):
        """add choropleth layer to map.

        Args:
            key (str): A selected value from tablemap's layer selection drop down menu.

        Returns:

        """

        # vmax_val = max(self.bldg_data_df[key])
        vmax_val = 1
        temp_id = list(range(len(self.inventory_df['guid'])))
        temp_id = [str(i) for i in temp_id]
        choro_data = dict(zip(temp_id, self.inventory_df[key]))
        try:
            self.map.remove_layer(self.layer)
            print("removed previous layer")
        except Exception:
            print("there is no existing layer")
            pass
        self.layer = ipylft.Choropleth(geo_data=self.inventory_json,
                                       choro_data=choro_data, colormap=linear.YlOrRd_04, value_min=0,
                                       value_max=vmax_val, border_color='black',
                                       style={'fillOpacity': 0.8}, name='dataset map')

        self.map.add_layer(self.layer)

        print('Done loading layer.')

    # TODO the following method for adding layer should be added in the future
    # def create_legend(self):
    #     legend = linear.YlOrRd_04.scale(0, self.vmax_val)
    #     self.tablemap.colormap = legend
    #     out = ipywgt.Output(layout={'border': '1px solid black'})
    #     with out:
    #         display(legend)
    #     widget_control = ipylft.WidgetControl(widget=out, position='topright')
    #     self.tablemap.add_control(widget_control)
    #     display(self.tablemap)
