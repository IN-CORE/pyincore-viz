

def get_period_and_demand_from_demandstr(demand_str: str):
    """Converts a demand type notation in string format to a dictionary of demandType and period (if applicable)

    Args:
        demand_str: Demand type represented in string format. Usually a combination of period and demand type
    wherever applicable. e.g. "PGA", "PGV", "0.2 sec SA", "0.2 SA".

    Returns: Dictionary of demandType and period

    """

    demand_parts = demand_str.split(" ")
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
