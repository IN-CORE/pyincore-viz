from typing import List


def get_period_and_demand_from_str(demand: str):
    """Converts a demand type notation in string format to a dictionary of demandType and period (if applicable)

    Args:
        demand: Demand type represented in string format. Usually a combination of period and demand type
    wherever applicable. e.g. "PGA", "PGV", "0.2 sec SA", "0.2 SA".

    Returns: Dictionary of demandType and period

    """

    demand_parts = demand.split(" ")
    if len(demand_parts) == 0:
        raise Exception("Invalid demand format")
    elif len(demand_parts) == 1:
        return {"demandType": demand_parts[0], "period": 0}
    else:
        try:
            return {"demandType": demand_parts[-1], "period": float(demand_parts[0])}
        except ValueError:
            print("Demand type provided is possibly not in the correct format")
            raise


def get_demands_for_dataset_hazards(datasets: List) -> List[str]:
    """Gets all the demands for the defined datasets of a dataset based hazard

    Args:
        datasets: List of datasets

    Returns: List of defined demands as strings

    """
    available_demands = []
    for dataset in datasets:
        available_demands.append(dataset['demandType'] if dataset['period'] == 0 else
                                 str(dataset['period']) + " " + dataset['demandType'])
    return available_demands
