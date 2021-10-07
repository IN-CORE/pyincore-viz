# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/
import importlib


class AnalysisViz:

    @staticmethod
    def visualize(dataset, **kwargs):
        """Base visualize method that dynamically imports the necessary modules.

            Args:
                dataset (obj): pyincore dataset without geospatial data.

            Returns:
                None

        """
        try:
            module_name = ""
            # split by namespace, capitalize then join
            for item in dataset.data_type.split(":"):
                module_name += item.capitalize()

            # load module
            # e.g. module_name = IncoreHousingUnitAllocation
            module = importlib.import_module("pyincore_viz.analysis." + module_name.lower())

            # load class
            analysis_class = getattr(module, module_name)

            # run vis
            return analysis_class.visualize(dataset, **kwargs)

        except Exception:
            raise ValueError("Fail to dynamically import dataset to its corresponding class. Please double "
                             "check the data_type of the dataset!")
