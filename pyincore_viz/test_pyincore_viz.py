from pyincore_viz import PlotUtil
from pyincore_viz import GeoUtil
from pyincore import IncoreClient, Dataset
from pyincore.analyses.bridgedamage import BridgeDamage
from pyincore.analyses.buildingdamage import BuildingDamage
from pyincore.analyses.meandamage import MeanDamage

import json, os

def run_with_base_class():
    client = IncoreClient()
    # client.login()

    # New madrid earthquake using Atkinson Boore 1995
    hazard_type = "earthquake"
    hazard_id = "5b902cb273c3371e1236b36b"

    # Use hazard uncertainty for computing damage
    use_hazard_uncertainty = False
    # Use liquefaction (LIQ) column of bridges to modify fragility curve
    use_liquefaction = False

    # Create bridge damage
    # NBSR bridges
    bridge_dataset_id = "5a284f2dc7d30d13bc082040"
    # Default Bridge Fragility Mapping on incore-service
    mapping_id = "5b47bcce337d4a37755e0cb2"
    bridge_dmg = BridgeDamage(client)
    # Load input datasets
    bridge_dmg.load_remote_input_dataset("bridges", bridge_dataset_id)
    # Specify the result name
    result_name = "bridge_result"
    # Set analysis parameters
    bridge_dmg.set_parameter("result_name", result_name)
    bridge_dmg.set_parameter("mapping_id", mapping_id)
    bridge_dmg.set_parameter("hazard_type", hazard_type)
    bridge_dmg.set_parameter("hazard_id", hazard_id)
    bridge_dmg.set_parameter("num_cpu", 1)

    bridge_inventory = bridge_dmg.get_input_dataset('bridges')

    # building inventory
    bldg_dmg = BuildingDamage(client)  # initializing pyincore
    bldg_dataset_id = "5df40388b9219c06cf8b0c80"  # defining building dataset (GIS point layer)
    bldg_dmg.load_remote_input_dataset("buildings", bldg_dataset_id)  # loading in the above
    mapping_id = "5d2789dbb9219c3c553c7977"  # specifiying mapping id from fragilites to building types
    bldg_dataset_id = "5df40388b9219c06cf8b0c80"  # defining building dataset (GIS point layer)
    bldg_dmg.load_remote_input_dataset("buildings", bldg_dataset_id)  # loading in the above
    building_dataset = bldg_dmg.get_input_dataset('buildings').get_inventory_reader()

    """
    histogram from csv
    """
    in_csv = 'C:\\rest\\output\\mc_output\\mc_failure_probability_buildings_eq_100yr.csv'
    plot_title = 'CSV histogram'
    x_title = 'x axis'
    y_title = 'y axix'
    column = 'failure_probability'
    num_bins = 30
    figure_size = (10, 5)
    fig = PlotUtil.histogram_from_csv_with_column(plot_title, x_title, y_title, column, in_csv, num_bins, figure_size)
    fig.show()

    """
    csv directory map
    """
    # inventory, column to map, and downloaded file path
    # bridge_dmg.run_analysis()
    # csv_dir_map = GeoUtil.map_csv_from_dir(bridge_inventory, column='hazardval')
    csv_dir = os.path.join('C:\\rest\\output', 'mc_output')
    csv_dir_map = GeoUtil.map_csv_from_dir(building_dataset, column='failure_probability', file_path=csv_dir)

    # Run bridge damage analysis
    # bridge_dmg.run_analysis()
    bridge_dmg = bridge_dmg.get_output_dataset('result')
    bldg_damge_df = bridge_dmg.get_dataframe_from_csv()
    inventory_df = PlotUtil.inventory_to_geodataframe(bridge_inventory)
    inventory_df = PlotUtil.remove_null_inventories(inventory_df, 'guid')
    inventory_df = PlotUtil.merge_inventory_w_dmg(inventory_df, bldg_damge_df)
    inventory_df.head()

    """
    geo map
    """
    new_map = PlotUtil.create_geo_map(inventory_df, key='hazardval')
    new_map

    new_map = PlotUtil.create_geo_map(inventory_df, key='ls-slight')
    new_map

    md = MeanDamage(client)
    md.set_input_dataset("damage", bridge_dmg)
    md.load_remote_input_dataset("dmg_ratios", "5a284f2cc7d30d13bc081f96")
    md.set_parameter("result_name", "bridge_mean_damage")
    md.set_parameter("damage_interval_keys",
                     ["none", "ds-slight", "ds-moderat", "ds-extensi",
                      "ds-complet"])
    md.set_parameter("num_cpu", 1)
    md.run_analysis()

    mean_damage_dataset = md.get_output_dataset('result')
    fig = PlotUtil.mean_damage_histogram(mean_damage_dataset, histogram_bins=30)

if __name__ == '__main__':
    run_with_base_class()