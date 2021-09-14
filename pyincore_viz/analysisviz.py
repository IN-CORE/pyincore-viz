import importlib


class AnalysisViz:

    @staticmethod
    def visualize(dataset):
        try:
            module_name = ""
            # split by namespace, capitalize then join
            for item in dataset.data_type.split(":"):
                module_name += item.capitalize()

            # load module
            # e.g. module_name = IncoreHousingUnitAllocation
            module = importlib.import_module("pyincore_viz.analysis." + module_name.lower())

            # load class
            analyse_class = getattr(module, module_name)

            # run vis
            analyse_class.vis()

        except:
            raise ValueError("Fail to dynamically import dataset to its corresponding class. Please double "
                             "check the data_type of the dataset!")
