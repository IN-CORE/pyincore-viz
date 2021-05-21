from pyincore_viz.helpers.common import get_period_and_demand_from_demandstr
import pytest


@pytest.mark.parametrize("demand_str,exp_demand_type,exp_period", [
    ("0.2 sec SA", "SA", 0.2),
    ("0.3 SA", "SA", 0.3),
    ("PGA", "PGA", 0)
])
def test_get_period_and_demand_from_demandstr(demand_str, exp_demand_type, exp_period):
    demand = get_period_and_demand_from_demandstr(demand_str)
    assert demand["demandType"] == exp_demand_type and demand["period"] == exp_period
