from typing import List, Tuple

from flwr.common import Metrics, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg


def aggregate_loss_by_weighted_average(evaluate_results: List[Tuple[ClientProxy, EvaluateRes]]) -> float:
    aggregated_loss = weighted_loss_avg([(evaluate_result.num_examples, evaluate_result.loss)
                                         for _, evaluate_result in evaluate_results])
    return aggregated_loss


def aggregate_metrics_by_weighted_average(metrics_tuples: list[tuple[int, Metrics]]) -> dict:
    # Initialize the aggregated metrics dictionary.
    aggregated_metrics = {}
    # Get the list of metrics names.
    metrics_names = list(metrics_tuples[0][1].keys())
    # Initialize the list that will store the results of multiplying metrics values by the number of examples,
    # for each participating client.
    metrics_products_list = []
    # Iterate through the list of metrics names.
    for metric_name in metrics_names:
        # For each pair (metric, participating client),
        # multiply the metric value by the number of examples used by the client (data contribution).
        metric_product = [num_examples * metric[metric_name] for num_examples, metric in metrics_tuples]
        metrics_products_list.append(metric_product)
    # Get the sum of examples used by all participating clients.
    sum_num_examples = sum([num_examples for num_examples, _ in metrics_tuples])
    # Aggregate the metrics by weighted average.
    for metric_index in range(0, len(metrics_names)):
        metric_name = metrics_names[metric_index]
        weighted_average_value = sum(metrics_products_list[metric_index]) / sum_num_examples
        aggregated_metrics.update({metric_name: weighted_average_value})
    return aggregated_metrics
