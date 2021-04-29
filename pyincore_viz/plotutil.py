# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from pyincore import StandardFragilityCurve, PeriodStandardFragilityCurve, PeriodBuildingFragilityCurve, \
    ConditionalStandardFragilityCurve, ParametricFragilityCurve, CustomExpressionFragilityCurve
from pyincore.models.fragilitycurverefactored import FragilityCurveRefactored
from pyincore.utils.expressioneval import Parser
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
        mean = numpy.log(numpy.square(location) /
                         numpy.sqrt(scale + numpy.square(location)))
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
    def get_standard_x_y(disttype: str, alpha: float, beta: float):
        """Get arrays of x and y values for standard fragility or period standard fragility

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
    def get_conditional_x_y(rules: dict, alpha_type, alpha: list, beta: list):
        x = numpy.linspace(0.001, 50, 200)
        y = []
        for i in x:
            index = ConditionalStandardFragilityCurve._fragility_curve_rules_match(rules, i)
            if index is not None:
                alpha_i = float(alpha[index])
                std_dev = 0
                beta_i = math.sqrt(math.pow(beta[index], 2) + math.pow(std_dev, 2))

                if alpha_type == 'median':
                    sp = (math.log(i) - math.log(alpha_i)) / beta_i
                    y.append(norm.cdf(sp))
                elif alpha_type == "lambda":
                    sp = (math.log(i) - alpha_i) / beta_i
                    y.append(norm.cdf(sp))
            else:
                raise ValueError("No matching rule has been found in this conditonal standard fragility curve. "
                                 "Please verify it's the right curve to use.")

        return x, y

    @staticmethod
    def get_period_building_x_y(a11_param, a12_param, a13_param, a14_param, a21_param, a22_param, period=0):
        # Assumption from Ergo BuildingLowPeriodSolver
        cutoff_period = 0.87

        x = numpy.linspace(0.001, 5, 200)
        if period < cutoff_period:
            multiplier = cutoff_period - period
            surface_eq = (numpy.log(x) - (cutoff_period * a12_param + a11_param)) / \
                         (a13_param + a14_param * cutoff_period)
            y = norm.cdf(surface_eq + multiplier * (numpy.log(x) - a21_param) / a22_param)
        else:
            y = norm.cdf(
                (numpy.log(x) - (a11_param + a12_param * period)) / (
                        a13_param + a14_param * period))

        return x, y

    @staticmethod
    def get_custom_x_y(expression):
        parser = Parser()
        x = numpy.linspace(0.001, 50, 200)
        y = []
        for i in x:
            variables = {'x': i}
            y.append(parser.parse(expression).evaluate(variables))

        return x, y

    @staticmethod
    def get_parametric_x_y(curve_type, parameters, **kwargs):
        if curve_type.lower() == "logit":
            y = numpy.linspace(0.001, 0.999, 200)
            cumulate_term = 0  # X*theta'
            A1 = 1  # coefficent for demand X

            for parameter_set in parameters:
                name = parameter_set["name"].lower()
                coefficient = parameter_set["coefficient"]
                default = parameter_set["interceptTermDefault"]
                if name == "demand":
                    A1 = 1 * coefficient
                else:
                    if name not in kwargs.keys():
                        cumulate_term += default * coefficient
                    else:
                        cumulate_term += kwargs[name] * coefficient
            x = numpy.exp((numpy.log(y / (1 - y)) - cumulate_term) / A1)

        else:
            raise ValueError("Other parametric functions than Logit has not been implemented yet!")

        return x, y

    @staticmethod
    def get_refactored_x_y(curve, demand_type_name, fragility_curve_parameters, custom_fragility_curve_parameters):
        x = numpy.linspace(0.001, 10, 200)
        y = []
        for i in x:
            y.append(curve.calculate_limit_state_probability(hazard_values={demand_type_name:i},
                                                             fragility_curve_parameters=fragility_curve_parameters,
                                                             **custom_fragility_curve_parameters)) #kwargs

        return x, y

    @staticmethod
    def get_refactored_x_y_z(curve, demand_type_names, fragility_curve_parameters, custom_fragility_curve_parameters):
        x = y = numpy.arange(1, 50, 0.5)

        def _f(curve, x, y):
            return curve.calculate_limit_state_probability(hazard_values={demand_type_names[0]: x,
                                                                          demand_type_names[1]: y},
                                                           fragility_curve_parameters= fragility_curve_parameters,
                                                           **custom_fragility_curve_parameters)

        X, Y = numpy.meshgrid(x, y)
        z = numpy.array([_f(curve, x, y) for x, y in zip(numpy.ravel(X), numpy.ravel(Y))])

        Z = z.reshape(X.shape)

        return X, Y, Z

    @staticmethod
    def get_fragility_plot(fragility_set, title=None):
        """Get fragility plot.

        Args:
            fragility_set (obj): A JSON like description of fragility assigned to the
                infrastructure inventory.

        Returns:
            collection: Plot and its style functions.

        """
        for curve in fragility_set.fragility_curves:
            if isinstance(curve, CustomExpressionFragilityCurve):
                if curve.expression.find('x') >= 0 and curve.expression.find('y') < 0:
                    x, y = PlotUtil.get_custom_x_y(curve.expression)
                else:
                    raise ValueError("We are only able to plot 2d fragility curve with x as variable name for now. "
                                     "More implementation coming soon...")

            elif isinstance(curve, StandardFragilityCurve) or isinstance(curve, PeriodStandardFragilityCurve):
                if curve.alpha_type == 'lambda':
                    alpha = curve.alpha
                elif curve.alpha_type == 'median':
                    alpha = math.log(curve.alpha)
                else:
                    raise ValueError("The alpha type is not implemented")
                x, y = PlotUtil.get_standard_x_y(
                    curve.curve_type, alpha, curve.beta)

            elif isinstance(curve, ConditionalStandardFragilityCurve):
                x, y = PlotUtil.get_conditional_x_y(curve.rules, curve.alpha_type, curve.alpha, curve.beta)

            elif isinstance(curve, ParametricFragilityCurve):
                x, y = PlotUtil.get_parametric_x_y(curve.curve_type, curve.parameters)

            elif isinstance(curve, PeriodBuildingFragilityCurve):
                x, y = PlotUtil.get_period_building_x_y(curve.fs_param0, curve.fs_param1, curve.fs_param2,
                                                        curve.fs_param3, curve.fs_param4, curve.fs_param5)
            else:
                raise ValueError("This type of fragility curve is not implemented!")

            plt.plot(x, y, label=curve.description)

        plt.xlabel((",").join(fragility_set.demand_types) + " (" + (",").join(fragility_set.demand_units) + ")")
        if title is None:
            title = fragility_set.description

        plt.title(title)
        plt.legend()

        return plt

    @staticmethod
    def get_fragility_plot_2d_refactored(fragility_set, title=None, custom_fragility_curve_parameters={}):
        demand_type_names = []
        for parameter in fragility_set.fragility_curve_parameters:
            # for  hazard
            if parameter.get("name") in fragility_set.demand_types or parameter.get("key") in \
                    fragility_set.demand_types:
                demand_type_names.append(parameter["name"])
            # check the rest of the parameters see if default or custom value has passed in
            else:
                if parameter.get("expression") is None and parameter.get("name") not in \
                        custom_fragility_curve_parameters:
                    raise ValueError("The required parameter: " + parameter.get("name")
                                     + " does not have a default or  custom value. Please check "
                                     "your fragility curve setting. Alternatively, you can include it in the "
                                     "custom_fragility_curve_parameters variable and passed it in this method. ")

        for curve in fragility_set.fragility_curves:
            x, y = PlotUtil.get_refactored_x_y(curve, demand_type_names[0],
                                               fragility_set.fragility_curve_parameters,
                                               custom_fragility_curve_parameters)
            plt.plot(x, y, label=curve.return_type["description"])

        plt.xlabel(fragility_set.demand_types[0] + " (" + fragility_set.demand_units[0] + ")")
        plt.title(title)
        plt.legend()

        return plt

    @staticmethod
    def get_fragility_plot_3d_refactored(fragility_set, title=None, limit_state="LS_0",
                                         custom_fragility_curve_parameters={}):
        demand_type_names = []
        for parameter in fragility_set.fragility_curve_parameters:
            # for  hazard
            if parameter.get("name") in fragility_set.demand_types or parameter.get("key") in \
                    fragility_set.demand_types:
                demand_type_names.append(parameter["name"])
            # check the rest of the parameters see if default or custom value has passed in
            else:
                if parameter.get("expression") is None and parameter.get("name") not in \
                        custom_fragility_curve_parameters:
                    raise ValueError("The required parameter: " + parameter.get("name")
                                     + " does not have a default or  custom value. Please check "
                                       "your fragility curve setting. Alternatively, you can include it in the "
                                       "custom_fragility_curve_parameters variable and passed it in this method. ")

        if len(demand_type_names) < 2:
            raise ValueError("This fragility curve set does not support 3D plot, please check if the number of demand "
                             "types are larger than 2.")

        # check if desired limit state exist, we can only plot one limit state per time for 3d plot
        matched = False
        for curve in fragility_set.fragility_curves:
            if limit_state == curve.return_type["description"]:
                matched = True
                x, y, z = PlotUtil.get_refactored_x_y_z(curve,
                                                        demand_type_names[:2],
                                                        fragility_set.fragility_curve_parameters,
                                                        custom_fragility_curve_parameters)
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
    def inventory_to_geodataframe(inventory_dataset):
        # TODO: need to move this method to Dataset Class
        fname = inventory_dataset.get_file_path(type="shp")
        inventory_df = gpd.read_file(fname)

        return inventory_df

    @staticmethod
    def remove_null_inventories(inventory_df, key='guid'):
        inventory_df.dropna(subset=[key], inplace=True)

        return inventory_df

    @staticmethod
    def dmg_state2value(damage_result, dmg_ratio_tbl):
        # TODO given damage ratio table, subtitute the damage state with actual mean damage factor
        pass

    @staticmethod
    def merge_inventory_w_dmg(inventory_df, damage_result):
        inventory_df = inventory_df.merge(damage_result, on='guid')

        return inventory_df

    @staticmethod
    def mean_damage_histogram(mean_damage_dataset, histogram_bins=20, figure_size=(10, 5), axes_font_size=12,
                              title_font_size=12):

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
        data_frame = pd.read_csv(in_csv)
        ax = data_frame[column].hist(bins=num_bins, figsize=figure_size)
        ax.set_title(plot_title)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)

        fig = ax.get_figure()

        return fig
