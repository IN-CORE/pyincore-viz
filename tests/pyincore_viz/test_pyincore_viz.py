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
