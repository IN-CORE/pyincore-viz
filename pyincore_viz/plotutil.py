# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy
import pandas as pd

from deprecated.sphinx import deprecated


class PlotUtil:
    """Plotting utility."""

    @deprecated(version="1.8.0", reason="It is not being used anymore. Check get_x_y or get_x_y_z")
    def get_standard_x_y(disttype: str, alpha: float, beta: float):
        """Get arrays of x and y values for standard fragility or period standard fragility.

        Args:
            disttype (str): A distribution type (log normal and normal).
            alpha (float): A distribution parameter (mostly mean).
            beta (float): A distribution parameter (mostly standard deviation).

        Returns:
            ndarray: X sampling values.
            ndarray: Y cumulative density values.

        """
        if disttype == 'LogNormal':
            return PlotUtil.sample_lognormal_cdf_alt(alpha, beta, 200)
        if disttype == 'Normal':
            return PlotUtil.sample_lognormal_cdf(alpha, beta, 200)
        if disttype == 'standardNormal':
            return PlotUtil.sample_normal_cdf(alpha, beta, 200)

    @staticmethod
    def get_x_y(curve, demand_type_name, curve_parameters, custom_curve_parameters,
                start=0.001, end=10, sample_size: int = 200):
        """Get arrays of x and y values for plotting refactored fragility curves.

        Args:
            curve (obj): An individual fragility curve object.
            demand_type_name (str): Demand type name
            curve_parameters (list): Default fragility curve parameters.
            start (float): A start value.
            end (float): An end value.
            sample_size (int): Number of points.
            **custom_curve_parameters: Keyword arguments.

        Returns:
            ndarray: X sampling values.
            ndarray: Y cumulative density values.

        """
        x = numpy.linspace(start, end, sample_size)
        y = []
        for i in x:
            y.append(curve.solve_curve_expression(hazard_values={demand_type_name: i},
                                                  curve_parameters=curve_parameters, **custom_curve_parameters))
        y = numpy.asarray(y)
        return x, y

    @staticmethod
    def get_x_y_z(curve, demand_type_names, curve_parameters, custom_curve_parameters, start=1, end=50,
                  sample_interval: int = 0.5):
        """Get arrays of x, y and z values for plotting refactored fragility plots.

        Args:
            curve (obj): An individual fragility curve object.
            demand_type_names (dict): Valid demand type names.
            curve_parameters (list): Default fragility curve parameters.
            **custom_curve_parameters: Keyword arguments.
            start (float): A start value.
            end (float): An end value.
            sample_interval (float): Sample interval.

        Returns:
            ndarray: X sampling values.
            ndarray: Y sampling values.
            ndarray: Z cumulative density values.

        """
        x = y = numpy.arange(start, end, sample_interval)

        def _f(curve, x, y):
            return curve.solve_curve_expression(hazard_values={demand_type_names[0]: x,
                                                demand_type_names[1]: y},
                                                curve_parameters=curve_parameters,
                                                **custom_curve_parameters)  # kwargs

        X, Y = numpy.meshgrid(x, y)
        z = numpy.array([_f(curve, x, y) for x, y in zip(numpy.ravel(X), numpy.ravel(Y))])

        Z = z.reshape(X.shape)

        return X, Y, Z

    @staticmethod
    def get_fragility_plot(fragility_set, title=None, dimension=2, limit_state="LS_0",
                           custom_curve_parameters={}, **kwargs):
        """Get fragility plot.

        Args:
            fragility_set (obj): A JSON like description of fragility assigned to the
                infrastructure inventory.
            title (str): A title of the plot.
            dimension (int): 2d versus 3d.
            limit_state (str): A limit state name, such as LS_0, or insignific, etc.
            custom_curve_parameters (dict): Custom fragility curve parameters.
                If you wish to overwrite default curve parameters (expression field).
            **kwargs: Keyword arguments.


        Returns:
            obj: Plot and its style functions.

        """
        ####################
        if title is None:
            title = fragility_set.description

        if dimension == 2:
            return PlotUtil.get_fragility_plot_2d(fragility_set, title, custom_curve_parameters, **kwargs)
        if dimension == 3:
            return PlotUtil.get_fragility_plot_3d(fragility_set, title, limit_state, custom_curve_parameters, **kwargs)
        else:
            raise ValueError("We do not support " + str(dimension) + "D fragility plotting")

    @staticmethod
    def get_fragility_plot_2d(fragility_set, title=None, custom_curve_parameters={}, **kwargs):
        """Get 2d refactored fragility plot.

        Args:
            fragility_set (obj): A JSON like description of fragility assigned to the
                infrastructure inventory.
            title (str): A title of the plot.
            custom_curve_parameters (dict): Custom fragility curve parameters.
                If you wish to overwrite default curve parameters (expression field).
            **kwargs: Keyword arguments.


        Returns:
            obj: Matplotlib pyplot object.

        """
        demand_type_names = []
        demand_types = fragility_set.demand_types
        for parameter in fragility_set.curve_parameters:
            # add case insensitive
            # for hazard
            if parameter.get("name") is not None \
                    and parameter.get("name").lower() \
                    in [demand_type.lower() for demand_type in demand_types]:
                demand_type_names.append(parameter["name"])
            elif parameter.get("fullName") is not None \
                    and parameter.get("fullName").lower() \
                    in [demand_type.lower() for demand_type in demand_types]:
                demand_type_names.append(parameter["fullName"])
            # check the rest of the parameters see if default or custom value has passed in
            else:
                if parameter.get("expression") is None and parameter.get("name") not in \
                        custom_curve_parameters:
                    raise ValueError("The required parameter: " + parameter.get("name")
                                     + " does not have a default or  custom value. Please check "
                                       "your fragility curve setting. Alternatively, you can include it in the "
                                       "custom_curve_parameters variable and passed it in this method. ")

        for curve in fragility_set.fragility_curves:
            x, y = PlotUtil.get_x_y(curve, demand_type_names[0], fragility_set.curve_parameters,
                                    custom_curve_parameters, **kwargs)
            plt.plot(x, y, label=curve.return_type["description"])

        plt.xlabel(fragility_set.demand_types[0] + " (" + fragility_set.demand_units[0] + ")")
        plt.title(title)
        plt.legend()

        return plt

    @staticmethod
    def get_fragility_plot_3d(fragility_set, title=None, limit_state="LS_0", custom_curve_parameters={}, **kwargs):
        """Get 3d refactored fragility plot.

        Args:
            fragility_set (obj): A JSON like description of fragility assigned to the
                infrastructure inventory.
            title (str): A title of the plot.
            limit_state (str): A limit state name, such as LS_0, or insignific, etc.
            custom_curve_parameters (dict): Custom fragility curve parameters.
                If you wish to overwrite default curve parameters (expression field).
            **kwargs: Keyword arguments.


        Returns:
            obj: Matplotlib pyplot object.

        """
        demand_type_names = []
        demand_types = fragility_set.demand_types
        for parameter in fragility_set.curve_parameters:
            # for hazard
            # add case insensitive
            if parameter.get("name") is not None \
                    and parameter.get("name").lower() \
                    in [demand_type.lower() for demand_type in demand_types]:
                demand_type_names.append(parameter["name"])
            elif parameter.get("fullName") is not None \
                    and parameter.get("fullName").lower() \
                    in [demand_type.lower() for demand_type in demand_types]:
                demand_type_names.append(parameter["fullName"])
            # check the rest of the parameters see if default or custom value has passed in
            else:
                if parameter.get("expression") is None and parameter.get("name") not in \
                        custom_curve_parameters:
                    raise ValueError("The required parameter: " + parameter.get("name")
                                     + " does not have a default or  custom value. Please check "
                                       "your fragility curve setting. Alternatively, you can include it in the "
                                       "custom_curve_parameters variable and passed it in this method. ")

        if len(demand_type_names) < 2:
            raise ValueError("This fragility curve set does not support 3D plot, please check if the number of demand "
                             "types are larger than 2.")

        # check if desired limit state exist, we can only plot one limit state per time for 3d plot
        matched = False
        for curve in fragility_set.fragility_curves:
            if limit_state == curve.return_type["description"]:
                matched = True
                x, y, z = PlotUtil.get_x_y_z(curve, demand_type_names[:2], fragility_set.curve_parameters,
                                             custom_curve_parameters, **kwargs)
                ax = plt.axes(projection='3d')
                ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
                ax.set_xlabel(fragility_set.demand_types[0] + " (" + fragility_set.demand_units[0] + ")")
                ax.set_ylabel(fragility_set.demand_types[1] + " (" + fragility_set.demand_units[1] + ")")
                ax.set_zlabel(limit_state + ' probability')

                plt.title(title)

        if not matched:
            raise ValueError("Limit State " + limit_state + " does not exist!")

        return plt

    @staticmethod
    @deprecated(version="1.9.0",
                reason="It is not being used anymore. Check pyincore's Dataset.get_dataframe_from_shapefile")
    def inventory_to_geodataframe(inventory_dataset):
        """Convert inventory_dataset to GeoDataFrame.

        Args:
            inventory_dataset (obj): An inventory dataset.

        Returns:
            GeoDataFrame: Inventory.

        """
        # TODO: need to move this method to Dataset Class
        fname = inventory_dataset.get_file_path(type="shp")
        inventory_df = gpd.read_file(fname)

        return inventory_df

    @staticmethod
    def remove_null_inventories(inventory_df, key='guid'):
        """Remove null inventory.

        Args:
            inventory_df (df): An inventory DataFrame.
            key (str): A key such as "guid".

        Returns:
            DataFrame: Inventory.

        """
        inventory_df.dropna(subset=[key], inplace=True)

        return inventory_df

    @staticmethod
    def dmg_state2value(damage_result, dmg_ratio_tbl):
        """Damage state to value.

        Args:
            damage_result (str): A damage result value.
            dmg_ratio_tbl (dict): A damage ratio table.

        Returns:

        """
        # TODO given damage ratio table, subtitute the damage state with actual mean damage factor
        pass

    @staticmethod
    def merge_inventory_w_dmg(inventory_df, damage_result):
        """Merge inventory with damages.

        Args:
            inventory_df (df): A Panda's Data frame inventory.
            damage_result (df): A Panda's Data frame damage results.

        Returns:
            DataFrame: Inventory.

        """
        inventory_df = inventory_df.merge(damage_result, on='guid')

        return inventory_df

    @staticmethod
    def mean_damage_histogram(mean_damage_dataset, histogram_bins=20, figure_size=(10, 5), axes_font_size=12,
                              title_font_size=12):
        """Figure with mean damage histogram.

        Args:
            mean_damage_dataset (obj): Mean damage dataset.
            histogram_bins (int): Number of bins.
            figure_size (list): Figure size, x and y.
            axes_font_size (int): Axle font size.
            title_font_size (int): Title font size.

        Returns:
            obj: Figure with damage histograms.

        """
        mean_damage = mean_damage_dataset.get_dataframe_from_csv()
        ax = mean_damage['meandamage'].hist(
            bins=histogram_bins, figsize=figure_size)
        ax.set_title("Mean damage distribution", fontsize=title_font_size)
        ax.set_xlabel("mean damage value", fontsize=axes_font_size)
        ax.set_ylabel("counts", fontsize=axes_font_size)
        fig = ax.get_figure()

        return fig

    @staticmethod
    def histogram_from_csv_with_column(plot_title, x_title, y_title, column, in_csv, num_bins, figure_size):
        """Get histogram from csv with column.

        Args:
            plot_title (str): A title of the plot.
            x_title (str): A title of the X axis.
            y_title (str): A title of the Y axis.
            column (str): A name of the column.
            in_csv (obj): A csv file with the column to be plotted.
            num_bins (int): Number of histogram bins.
            figure_size (list): Figure size, x and y.

        Returns:
            obj: Figure with damage histograms.

        """
        data_frame = pd.read_csv(in_csv)
        ax = data_frame[column].hist(bins=num_bins, figsize=figure_size)
        ax.set_title(plot_title)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)

        fig = ax.get_figure()

        return fig
