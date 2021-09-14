import importlib

class AnalysisViz:

    @staticmethod
    def visualize(dataset):
        # module_name = dataset.data_type
        module_name = "IncoreHousingUnitAllocation"
        module = importlib.import_module("analyse." + module_name.lower())

        dataset = getattr(module, module_name)
        dataset.vis()


if __name__ == "__main__":
    AnalysisViz.visualize(dataset=None)