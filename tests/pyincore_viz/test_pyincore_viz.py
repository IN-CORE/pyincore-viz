import pytest
import matplotlib
import os
import jwt

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


@pytest.fixture
def client(monkeypatch):
    try:
        with open(os.path.join(os.path.dirname(__file__), ".incorepw"), 'r') as f:
            cred = f.read().splitlines()
    except EnvironmentError:
        assert False
    credentials = jwt.decode(cred[0], cred[1])
    monkeypatch.setattr("builtins.input", lambda x: credentials["username"])
    monkeypatch.setattr("getpass.getpass", lambda y: credentials["password"])
    client = IncoreClient(service_url=INCORE_API_DEV_URL, token_file_name=".incrtesttoken")

    return client


def test_visualize_earthquake(client):
    eq_hazard_id = "5b902cb273c3371e1236b36b"

    eq_metadata = HazardService(client).get_earthquake_hazard_metadata(eq_hazard_id)
    eq_dataset_id = eq_metadata['rasterDataset']['datasetId']

    eq_dataset = Dataset.from_data_service(eq_dataset_id, DataService(client))
    viz.plot_earthquake(eq_hazard_id, client)

    assert True


def test_visualize_joplin_tornado_building(client):
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

    assert True


def test_visualize_inventory(client):
    shelby_hospital_inv_id = "5a284f0bc7d30d13bc081a28"
    shelby_road_id = "5a284f2bc7d30d13bc081eb6"

    # get shelvy building inventory and road
    sh_bldg_inv = Dataset.from_data_service(shelby_hospital_inv_id, DataService(client))
    sh_road = Dataset.from_data_service(shelby_road_id, DataService(client))

    # visualize building inventory
    viz.plot_map(sh_bldg_inv, column="struct_typ", category=False, basemap=True)

    # visualize building inventory from geoserver
    viz.get_wms_map([sh_bldg_inv, sh_road], zoom_level=10)
    viz.get_gdf_map([sh_bldg_inv, sh_road], zoom_level=10)
    viz.get_gdf_wms_map([sh_bldg_inv], [sh_road], zoom_level=10)

    assert True


def test_visualize_network(client):
    centerville_epn_network_id = "5d25fb355648c40482a80e1c"

    dataset = Dataset.from_data_service(centerville_epn_network_id, DataService(client))
    network_dataset = NetworkDataset(dataset)
    viz.plot_network_dataset(network_dataset, 12)

    assert True


def test_plot_fragility(client):
    # clean plots that are not closed/cleared
    matplotlib.pyplot.clf()
    matplotlib.pyplot.cla()

    # 5b47b2d7337d4a36187c61c9 period standard
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("5b47b2d7337d4a36187c61c9"))
    plt = plot.get_fragility_plot(fragility_set, title="period standard fragility curve")
    plt.savefig('periodStandard.png')
    plt.clf()

    # 5b4903c7337d4a48f7d88dcf standard
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("5b4903c7337d4a48f7d88dcf"))
    plt = plot.get_fragility_plot(fragility_set, title="standard fragility curve")
    plt.savefig('standard.png')
    plt.clf()

    # 5b47b34e337d4a36290754a0 period building
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("5b47b34e337d4a36290754a0"))
    plt = plot.get_fragility_plot(fragility_set, title="period building fragility curve")
    plt.savefig('periodBuilding.png')
    plt.clf()

    # 5ed6bfc35b6166000155d0d9 parametric
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("5ed6bfc35b6166000155d0d9"))
    plt = plot.get_fragility_plot(fragility_set, title="parametric fragility curve")
    plt.savefig('parametric.png')
    plt.clf()

    # 5b47ba6f337d4a372105936f custom 2d
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("5b47ba6f337d4a372105936f"))
    plt = plot.get_fragility_plot(fragility_set, title="customExpression 2d fragility curve")
    plt.savefig('customExpression.png')
    plt.clf()

    # 5ed6be9a5b6166000155d0b9 custom 2d
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("5ed6be9a5b6166000155d0b9"))
    plt = plot.get_fragility_plot(fragility_set, title="conditional fragility curve")
    plt.savefig('conditional.png')
    plt.clf()

    assert True


def test_visualize_raster_file(client):
    galvaston_wave_height_id = '5f11e503feef2d758c4df6db'
    dataset = Dataset.from_data_service(galvaston_wave_height_id, DataService(client))
    map = viz.plot_raster_from_path(dataset.get_file_path('tif'))

    assert True


def test_plot_map_dataset_list(client):
    galveston_roadway_id = '5f0dd5ecb922f96f4e962caf'
    galvaston_wave_height_id = '5f11e503feef2d758c4df6db'
    shelvy_building_damage_id = '5a296b53c7d30d4af5378cd5'
    dataset_id_list = [galveston_roadway_id, galvaston_wave_height_id, shelvy_building_damage_id]

    dataset_list = []
    for dataset_id in dataset_id_list:
        dataset_list.append(Dataset.from_data_service(dataset_id, DataService(client)))

    # table dataset plot map
    map = viz.plot_maps_dataset_list(dataset_list, client)

    assert True


def test_plot_map_table_dataset(client):
    building_damage_id = '5a296b53c7d30d4af5378cd5'
    dataset = Dataset.from_data_service(building_damage_id, DataService(client))
    map = viz.plot_table_dataset(dataset, client, 'meandamage')

    assert True


def test_plot_table_dataset_list_from_single_source(client):
    seaside_building_polygon_id = '5f7c95d681c8dd4d309d5a46'
    dataset_id_list = ['5f7c9b4f81c8dd4d309d5b62', '5f7c9af781c8dd4d309d5b5e']
    dataset_list = []

    for dataset_id in dataset_id_list:
        dataset_list.append(Dataset.from_data_service(dataset_id, DataService(client)))

    map = viz.plot_table_dataset_list_from_single_source(
        client, dataset_list, 'failure_probability', seaside_building_polygon_id)

    assert True
