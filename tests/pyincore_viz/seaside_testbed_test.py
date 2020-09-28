from pyincore import IncoreClient, Dataset, FragilityService, MappingSet
from pyincore.analyses.buildingdamage import BuildingDamage
from pyincore.analyses.cumulativebuildingdamage import CumulativeBuildingDamage
from pyincore.analyses.montecarlofailureprobability import MonteCarloFailureProbability
from pyincore_viz.geoutil import GeoUtil as viz

import os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

if __name__ == "__main__":

    geotiff_path = 'C:\\Users\\ywkim\\Documents\\NIST\\Galveston\\Hurricane\\Surge_Raster.tif'
    map = viz.plot_raster_from_path(geotiff_path)

    client = IncoreClient()

    hazard_type = "earthquake"
    # rt = [100, 250, 500, 1000, 2500, 5000, 10000]
    # rt_hazard_dict = {100: "5dfa4058b9219c934b64d495",
    #                   250: "5dfa41aab9219c934b64d4b2",
    #                   500: "5dfa4300b9219c934b64d4d0",
    #                   1000: "5dfa3e36b9219c934b64c231",
    #                   2500: "5dfa4417b9219c934b64d4d3",
    #                   5000: "5dfbca0cb9219c101fd8a58d",
    #                  10000: "5dfa51bfb9219c934b68e6c2"}
    rt = [100]
    rt_hazard_dict = {100: "5dfa4058b9219c934b64d495"}

    bldg_eq_dmg_result_list = []  # place holder to saving earthquake building damage result iteration

    for rt_val in rt:  # loop through recurrence interval
        bldg_dmg = BuildingDamage(client)  # initializing pyincore
        bldg_dataset_id = "5df40388b9219c06cf8b0c80"  # defining building dataset (GIS point layer)
        bldg_dmg.load_remote_input_dataset("buildings", bldg_dataset_id)  # loading in the above
        mapping_id = "5d2789dbb9219c3c553c7977"  # specifiying mapping id from fragilites to building types
        fragility_service = FragilityService(client)  # loading fragility mapping
        mapping_set = MappingSet(fragility_service.get_mapping(mapping_id))
        bldg_dmg.set_input_dataset("dfr3_mapping_set", mapping_set)

        result_name = 'buildings_eq_' + str(rt_val) + 'yr_dmg_result'  # defining output name

        bldg_dmg.set_parameter("hazard_type", hazard_type)  # defining hazard type (e.g. earthquake vs. tsunami)
        hazard_id = rt_hazard_dict[rt_val]  # specifying hazard id for specific recurrence interval
        bldg_dmg.set_parameter("hazard_id", hazard_id)  # loading above into pyincore
        bldg_dmg.set_parameter("num_cpu", 4)  # number of CPUs to use for parallel processing
        bldg_dmg.set_parameter("result_name", result_name)  # specifying output name in pyincore

        bldg_dmg.run_analysis()  # running the analysis with the above parameters
        bldg_eq_dmg_result_list.append(bldg_dmg.get_output_dataset('result'))

    print("Test")

    m = viz.plot_table_dataset(client, bldg_eq_dmg_result_list, 'failure_probability', '5d927ab2b9219c06ae8d313c')

    # bldg_inventory_id = '5d927ab2b9219c06ae8d313c' # polygons
    m  # showing maps in notebook
