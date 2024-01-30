# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/
import json

import pytest
import matplotlib
import geopandas as gpd

from pyincore import Dataset
from pyincore import NetworkDataset
from pyincore.dataservice import DataService
from pyincore.fragilityservice import FragilityService
from pyincore.hazardservice import HazardService
from pyincore.models.fragilitycurveset import FragilityCurveSet
from pyincore_viz.gistutil import GeoUtil as viz
from pyincore_viz.plotutil import PlotUtil as plot


@pytest.fixture
def client():
    return pytest.client


def test_visualize_model_earthquake(client):
    eq_hazard_id = "5b902cb273c3371e1236b36b"
    viz.plot_earthquake(eq_hazard_id, client)
    assert True


def test_visualize_dataset_earthquake(client):
    eq_hazard_id = "5ba8ed5cec23090435209069"
    viz.plot_earthquake(eq_hazard_id, client)
    assert True


def test_visualize_dataset_earthquake_with_demand(client):
    eq_hazard_id = "5ba8ed5cec23090435209069"
    viz.plot_earthquake(eq_hazard_id, client, "0.4 SA")
    assert True


def test_visualize_joplin_tornado_building(client):
    # testing datasets
    tornado_hazard_id = "5dfa32bbc0601200080893fb"
    joplin_bldg_inv_id = "5df7d0de425e0b00092d0082"

    viz.plot_tornado(tornado_hazard_id, client, basemap=False)

    tornado_dataset_id = HazardService(client).get_tornado_hazard_metadata(tornado_hazard_id)[
        'hazardDatasets'][0].get('datasetId')
    tornado_dataset = Dataset.from_data_service(tornado_dataset_id, DataService(client))

    viz.get_gdf_map([tornado_dataset])

    # get joplin building inventory
    joplin_bldg_inv = Dataset.from_data_service(joplin_bldg_inv_id, DataService(client))

    # using wms layer for joplin building inv. gdf will crash the browser
    viz.get_gdf_wms_map([tornado_dataset], [joplin_bldg_inv])

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
    viz.get_wms_map([sh_bldg_inv, sh_road])
    viz.get_gdf_map([sh_bldg_inv, sh_road])
    viz.get_gdf_wms_map([sh_bldg_inv], [sh_road])

    assert True


def test_visualize_network(client):
    centerville_epn_network_id = "5d25fb355648c40482a80e1c"

    dataset = Dataset.from_data_service(centerville_epn_network_id, DataService(client))
    network_dataset = NetworkDataset(dataset)
    viz.plot_network_dataset(network_dataset)

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

    # 5ed6be9a5b6166000155d0b9 conditional 2d
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("5ed6be9a5b6166000155d0b9"))
    plt = plot.get_fragility_plot(fragility_set, title="conditional fragility curve")
    plt.savefig('conditional.png')
    plt.clf()

    # new format 2d
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("602f31f381bd2c09ad8efcb4"))
    # comment on and off to compare curves
    # plt = plot.get_fragility_plot_2d(fragility_set, title="refactored fragility 2d curve")
    plt = plot.get_fragility_plot_2d(fragility_set, title="refactored fragility 2d curve",
                                     custom_curve_parameters={"ffe_elev": 3})
    # you can now also plot refactored fragility curve using the main plot method
    # plt = plot.get_fragility_plot(fragility_set, title="refactored fragility 2d curve",
    #                               custom_curve_parameters={"ffe_elev": 3})

    plt.savefig('refactored_2d.png')
    plt.clf()

    # new format 3d
    fragility_set = FragilityCurveSet(FragilityService(client).get_dfr3_set("5f6ccf67de7b566bb71b202d"))
    plt = plot.get_fragility_plot_3d(fragility_set, title="refactored fragility 3d curve", limit_state="LS_0")
    # you can now also plot refactored fragility curve using the main plot method
    # plt = plot.get_fragility_plot(fragility_set, title="refactored fragility 3d curve", limit_state="LS_0",
    #                               dimension=3, custom_curve_parameters={"ffe_elev": 3})
    plt.savefig('refactored_3d.png')
    plt.clf()

    # test case sensitivity of demand types
    import pathlib, os
    working_dir = pathlib.Path(__file__).parent.resolve()
    fragility_set = FragilityCurveSet.from_json_file(
        os.path.join(working_dir, "data", "StandardFragilityCurveDemandType.json"))
    plt = plot.get_fragility_plot_2d(fragility_set, title="demand type case insensitive fragility 2d curve")
    plt.savefig('case_insensitive_2d.png')
    plt.clf()

    assert True


def test_plot_raster_dataset(client):
    galveston_deterministic_hurricane = "5f10837ab922f96f4e9ffb86"
    viz.plot_raster_dataset(galveston_deterministic_hurricane, client)

    assert True


def test_visualize_raster_file(client):
    galvaston_wave_height_id = '5f11e503feef2d758c4df6db'
    dataset = Dataset.from_data_service(galvaston_wave_height_id, DataService(client))
    map = viz.map_raster_overlay_from_file(dataset.get_file_path('tif'))

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


def test_heatmap(client):
    shelby_hospital_inv_id = "5a284f0bc7d30d13bc081a28"
    dataset = Dataset.from_data_service(shelby_hospital_inv_id, DataService(client))
    map = viz.plot_heatmap(dataset, "str_prob")

    assert True


def test_seaside_bridges(client):
    trns_brdg_dataset_id = "5d251172b9219c0692cd7523"
    trns_brdg_dataset = Dataset.from_data_service(trns_brdg_dataset_id, DataService(client))
    viz.plot_map(trns_brdg_dataset, column=None, category=False, basemap=True)

    assert True


def test_overay_gdf_with_raster(client):
    shelby_hospital_inv_id = "5a284f0bc7d30d13bc081a28"
    shelby_census_tract = "5a284f4cc7d30d13bc0822d4"
    memphis_water_pipeline = "5a284f28c7d30d13bc081d14"
    memphis_eq = "5b902cb273c3371e1236b36b"

    eq_dataset_id = HazardService(client).get_earthquake_hazard_metadata(memphis_eq)['hazardDatasets'][0].get('datasetId')
    raster_dataset = Dataset.from_data_service(eq_dataset_id, DataService(client))

    dataset = Dataset.from_data_service(shelby_hospital_inv_id, DataService(client))
    gdf = gpd.read_file(dataset.local_file_path)

    map = viz.overlay_gdf_with_raster_hazard(gdf, "struct_typ", raster_dataset)

    assert True


def test_choropleth_sinlge_dataset(client):
    social_vulnerability_census_block_group = '5a284f57c7d30d13bc08254c'
    dataset = Dataset.from_data_service(social_vulnerability_census_block_group, DataService(client))
    viz.plot_choropleth_multiple_fields_from_single_dataset(dataset, ['tot_hh', 'totpop'])

    assert True


def test_choropleth_multiple_dataset(client):
    social_vulnerability_census_block_group = '5a284f57c7d30d13bc08254c'
    dislocation_census_block_group = '5a284f58c7d30d13bc082566'
    dataset1 = Dataset.from_data_service(social_vulnerability_census_block_group, DataService(client))
    dataset2 = Dataset.from_data_service(dislocation_census_block_group, DataService(client))
    viz.plot_choropleth_multiple_dataset([dataset1, dataset2], ['tot_hh', 'p_16pyr'])

    assert True


def test_multiple_vector_visualization(client):
    centerville_model_tornado = '60c917b498a93232884f367d'
    centerville_epn_link = '5b1fdc2db1cf3e336d7cecc9'
    tornado_metadata = HazardService(client).get_tornado_hazard_metadata(centerville_model_tornado)
    dataset1 = Dataset.from_data_service(centerville_epn_link, DataService(client))
    dataset2 = Dataset.from_data_service(tornado_metadata["hazardDatasets"][0].get("datasetId"),
                                         DataService(client))
    viz.plot_multiple_vector_dataset([dataset1, dataset2])

    assert True
