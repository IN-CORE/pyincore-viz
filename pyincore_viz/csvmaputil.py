# Copyright (c) 2020 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import os
import json
import ipyleaflet as ipylft
import ipywidgets as ipywgt
import pandas as pd

from branca.colormap import linear
from pyincore_viz.plotutil import PlotUtil
from pyincore_viz import globals

logger = globals.LOGGER


class CsvMapUtil:
    """Utility methods for creating csv directory based map"""
    csvmap = None

    @staticmethod
    def generate_map_csv_from_dir(inventory_dataset, column, file_path=None):
        """Creates map window with given inventory with multiple csv files using folder location

        Args:
            inventory_dataset (Dataset):  pyincore inventory Dataset object
            column (str): column name to use for the mapping visualization
            file_path (str): file path that contains csv files

        Returns:
            csvmap (ipyleaflet.Map): ipyleaflet Map object

        """

        inventory_df = PlotUtil.inventory_to_geodataframe(inventory_dataset)
        inventory_df = PlotUtil.remove_null_inventories(inventory_df, 'guid')

        CsvMapUtil.csvmap = CsvMapUtil.create_basemap_ipylft(inventory_df)

        if file_path is None:
            file_path = os.getcwd()
        data, outfiles = CsvMapUtil.load_all_data(file_path, column)
        CsvMapUtil.inventory_df = CsvMapUtil.merge_inventory_data(data, inventory_df)
        CsvMapUtil.inventory_json = json.loads(CsvMapUtil.inventory_df.to_json())
        CsvMapUtil.create_map_widgets(outfiles)

        return CsvMapUtil.csvmap

    def create_basemap_ipylft(geo_dataframe):
        """Creates map window with given inventory with multiple csv file using folder location

        Args:
            geo_dataframe (DataFrame): Geopandas DataFrame object

        Returns:
            m(ipyleaflet.Map): ipyleaflet Map object

        """

        ext = geo_dataframe.total_bounds
        cen_x, cen_y = (ext[1] + ext[3]) / 2, (ext[0] + ext[2]) / 2
        m = ipylft.Map(center=(cen_x, cen_y), zoom=12, basemap=ipylft.basemaps.Stamen.Toner, scroll_wheel_zoom=True)

        return m

    def load_all_data(path_to_data, column_name):
        """Loading in all data in output path

        Args:
            path_to_data (str): path name contains csv files
            column_name (str): column name to use for the mapping visualization

        Returns:
            data (DataFrame): Pandas DataFrame contains all the csv file content
            outfiles (list): list of the file names in the folder

        """

        temp_outfiles = os.listdir(path_to_data)
        outfiles = []
        for temp_outfile in temp_outfiles:
            file_root, file_extension = os.path.splitext(temp_outfile)
            if file_extension.lower() == '.csv':
                filename = os.path.join(path_to_data, temp_outfile)
                data = pd.read_csv(filename, dtype=str)
                try:
                    data[temp_outfile] = data[column_name].astype(float)
                    outfiles.append(temp_outfile)
                except KeyError as err:
                    logger.debug("Skipping " + filename +
                                 ", Given column name does not exist or the column is not number.")

        csv_index = 0
        data = None

        if len(outfiles) == 0:
            print("There is no csv files with give field with numeric value.")
            exit(1)

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
                data = data[['guid', file]]
            else:
                temp = pd.read_csv(filename, dtype=str)
                temp[file] = temp[column_name].astype(float)
                temp = temp[['guid', file]]
                data = data.merge(temp, on='guid')
            csv_index += 1

        return data, outfiles

    def merge_inventory_data(data, data_df):
        """Merge two DataFrames as one

            Args:
                data (DataFrame): Pandas DataFrame contains the csv file content
                data_df (DataFrame): Pandas DataFrame contains the csv file content

            Returns:
                data_df (DataFrame): Pandas DataFrame contains the  merged two DataFrames

            """

        data_df = data_df.merge(data, on='guid')

        return data_df

    def create_map_widgets(outfiles):
        """Create and add map widgets into csv map

        Args:
            outfiles (list): list of the file names in the folder

        """
        CsvMapUtil.csv_dir_map_dropdown = ipywgt.Dropdown(description='Outputfile - 1', options=outfiles, width=500)
        file_control1 = ipylft.WidgetControl(widget=CsvMapUtil.csv_dir_map_dropdown, position='bottomleft')

        # use the following line when it needs to have another dropdown
        # self.dropdown2 = ipywgt.Dropdown(description = 'Outputfile - 2', options = outfiles, width=500)
        # file_control2 = ipylft.WidgetControl(widget=self.dropdown2, position='bottomleft')

        button = ipywgt.Button(description='Generate Map', button_style='info')
        button.on_click(CsvMapUtil.on_button_clicked)
        map_control = ipylft.WidgetControl(widget=button, position='bottomleft')

        CsvMapUtil.csvmap.add_control(ipylft.LayersControl(position='topright', style='info'))
        CsvMapUtil.csvmap.add_control(ipylft.FullScreenControl(position='topright'))
        CsvMapUtil.csvmap.add_control(map_control)
        # CsvMapUtil.csvmap.add_control(file_control2)      # use the line when it needs to have extra dropdown
        CsvMapUtil.csvmap.add_control(file_control1)

    def on_button_clicked(b):
        """button click action for csv map

        Args:
            b (action): button click action for csvmap

        """
        # def on_button_clicked(b, csv_dir_map_dropdown, inventory_df, inventory_json):
        print('Loading: ', CsvMapUtil.csv_dir_map_dropdown.value)
        key = CsvMapUtil.csv_dir_map_dropdown.value
        CsvMapUtil.create_choropleth_layer(key)
        print('\n')

    def create_choropleth_layer(key):
        """add choropleth layer to csv map

        Args:
            key (str): selected value from csvmap's layer selection drop down menu

        """

        # vmax_val = max(self.bldg_data_df[key])
        vmax_val = 1
        temp_id = list(range(len(CsvMapUtil.inventory_df['guid'])))
        temp_id = [str(i) for i in temp_id]
        choro_data = dict(zip(temp_id, CsvMapUtil.inventory_df[key]))
        layer = ipylft.Choropleth(geo_data=CsvMapUtil.inventory_json, choro_data=choro_data, colormap=linear.YlOrRd_04,
                                  value_min=0, value_max=vmax_val, border_color='black', style={'fillOpacity': 0.8},
                                  name='CSV map')
        CsvMapUtil.csvmap.add_layer(layer)

        print('Done loading layer.')

    # TODO the following method for adding layer should be added in the future
    # def create_legend(self):
    #     legend = linear.YlOrRd_04.scale(0, self.vmax_val)
    #     CsvMapUtil.csvmap.colormap = legend
    #     out = ipywgt.Output(layout={'border': '1px solid black'})
    #     with out:
    #         display(legend)
    #     widget_control = ipylft.WidgetControl(widget=out, position='topright')
    #     CsvMapUtil.csvmap.add_control(widget_control)
    #     display(CsvMapUtil.csvmap)
