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
    # Get the sum of examples used by all participating clients.
    sum_num_examples = sum([num_examples for num_examples, _ in metrics_tuples])
    # Get the list of metrics names.
    metrics_names = []
    for _, metrics in metrics_tuples:
        for metric_key, _ in metrics.items():
            metrics_names.append(metric_key)
    metrics_names = sorted(list(set(metrics_names)))
    # Iterate through the list of metrics names.
    for metric_name in metrics_names:
        # Initialize the 'sum_metric_product' variable.
        sum_metric_product = 0
        # For each pair (metric, participating client)...
        for num_examples, metrics in metrics_tuples:
            # If the current client metrics contains the current metric...
            if metric_name in metrics:
                # Multiply the metric value by the number of examples used by the client (data contribution),
                # and increment it into the 'sum_metric_product' variable.
                metric_value = metrics[metric_name]
                sum_metric_product += num_examples * metric_value
        # Get the weighted average value of the current metric.
        metric_weighted_average_value = sum_metric_product / sum_num_examples
        # Append it into the aggregated metrics dictionary.
        aggregated_metrics.update({metric_name: metric_weighted_average_value})
    # Return the aggregated metrics dictionary.
    return aggregated_metrics
