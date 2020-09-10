import os

from pyincore import Dataset
from pyincore import IncoreClient
from pyincore import NetworkDataset
from pyincore.dataservice import DataService
from pyincore.fragilityservice import FragilityService
from pyincore.globals import INCORE_API_DEV_URL
from pyincore.hazardservice import HazardService
from pyincore.models.fragilitycurveset import FragilityCurveSet

from pyincore_viz.geoutil import GeoUtil as viz
from pyincore_viz.plotutil import PlotUtil as plot

client = IncoreClient(INCORE_API_DEV_URL)


def test_visualize_earthquake():
    eq_hazard_id = "5b902cb273c3371e1236b36b"

    eq_metadata = HazardService(client).get_earthquake_hazard_metadata(eq_hazard_id)
    eq_dataset_id = eq_metadata['rasterDataset']['datasetId']

    eq_dataset = Dataset.from_data_service(eq_dataset_id, DataService(client))
    viz.plot_earthquake(eq_hazard_id, client)


def test_visualize_joplin_tornado_building():
    # testing datasets
    tornado_hazard_id = "5dfa32bbc0601200080893fb"
    joplin_bldg_inv_id = "5df7d0de425e0b00092d0082"

    viz.plot_tornado(tornado_hazard_id, client, basemap=False)

    tornado_dataset_id = HazardService(client).get_tornado_hazard_metadata(tornado_hazard_id)['datasetId']
    tornado_dataset = Dataset.from_data_service(tornado_dataset_id, DataService(client))

    viz.get_gdf_map([tornado_dataset], zoom_level=11)

    # get joplin building inventory
    joplin_bldg_inv = Dataset.from_data_service(joplin_bldg_inv_id, DataService(client))

    # using wms layer for joplin building inv. gdf will crash the browser
    viz.get_gdf_wms_map([tornado_dataset], [joplin_bldg_inv], zoom_level=11)


def test_visualize_inventory():
    shelby_hopital_inv_id = "5a284f0bc7d30d13bc081a28"
    shelby_road_id = "5a284f2bc7d30d13bc081eb6"

    # get shelvy building inventory and road
    sh_bldg_inv = Dataset.from_data_service(shelby_hopital_inv_id, DataService(client))
    sh_road = Dataset.from_data_service(shelby_road_id, DataService(client))

    # visualize building inventory
    viz.plot_map(sh_bldg_inv, column="struct_typ", category=False, basemap=True)

    # visualize building inventory from geoserver
    viz.get_wms_map([sh_bldg_inv, sh_road], zoom_level=10)
    viz.get_gdf_map([sh_bldg_inv, sh_road], zoom_level=10)
    viz.get_gdf_wms_map([sh_bldg_inv], [sh_road], zoom_level=10)


def test_visualize_network():
    centerville_epn_network_id = "5d25fb355648c40482a80e1c"

    dataset = Dataset.from_data_service(centerville_epn_network_id, DataService(client))
    network_dataset = NetworkDataset(dataset)
    viz.plot_network_dataset(network_dataset, 12)


def test_map_csv():
    seaside_bldg_id = "5f2b1354f3e24203f9f60026"  # defining building dataset (GIS point layer)
    seaside_bldg_inv = Dataset.from_data_service(seaside_bldg_id, DataService(client))
    csv_dir = os.path.join("examples", "seaside_bldg_dmg_output_csv")
    csv_dir_map = viz.map_csv_from_dir(seaside_bldg_inv, column='failure_probability', file_path=csv_dir)


def test_plot_fragility():
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("5b47b34e337d4a36290754a0"))
    plt = plot.get_fragility_plot(fragility_set)
    plt.savefig('test.png')


if __name__ == "__main__":
    # comment out or remove comment to test specific feature below.
    # test_visualize_earthquake()
    # test_visualize_joplin_tornado_building()
    # test_visualize_inventory()
    # test_visualize_network()
    # test_map_csv()
    test_plot_fragility()