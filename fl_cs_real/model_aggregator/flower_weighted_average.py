from typing import List, Tuple

from flwr.common import FitRes, ndarrays_to_parameters, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace


def aggregate_parameters_by_weighted_average(fit_results: List[Tuple[ClientProxy, FitRes]],
                                             inplace_aggregation: bool) -> Parameters:
    if inplace_aggregation:
        aggregated_ndarrays = aggregate_inplace(fit_results)
    else:
        weights_results = [(parameters_to_ndarrays(fit_result.parameters), fit_result.num_examples)
                           for _, fit_result in fit_results]
        aggregated_ndarrays = aggregate(weights_results)
    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
    return parameters_aggregated
