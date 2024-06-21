# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/
import pytest

from pyincore.analyses.housingunitallocation.housingunitallocation import (
    HousingUnitAllocation,
)
from pyincore_viz import AnalysisViz


@pytest.fixture
def client():
    return pytest.client


def test_hua_visualization(client):
    # Joplin
    building_inv = "5f218e36114b783cb0b01833"
    housing_unit_inv = "5df7ce61425e0b00092d0013"  # 2ev3
    address_point_inv = "5df7ce0d425e0b00092cffee"  # 2ev2
    result_name = "IN-CORE_2ev3_HUA"

    seed = 1238
    iterations = 2

    spa = HousingUnitAllocation(client)
    spa.load_remote_input_dataset("buildings", building_inv)
    spa.load_remote_input_dataset("housing_unit_inventory", housing_unit_inv)
    spa.load_remote_input_dataset("address_point_inventory", address_point_inv)

    spa.set_parameter("result_name", result_name)
    spa.set_parameter("seed", seed)
    spa.set_parameter("iterations", iterations)

    spa.run_analysis()

    # test viz
    dataset = spa.get_output_dataset("result")
    df = AnalysisViz.visualize(dataset)

    assert df is not None
