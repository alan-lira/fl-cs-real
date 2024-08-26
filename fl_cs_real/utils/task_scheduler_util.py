from numpy import any, arange, ndarray, max, sum


def get_num_selected_resources(assignment: ndarray) -> int:
    num_selected_resources = len([index for index, value in enumerate(list(assignment)) if value > 0])
    return num_selected_resources


def get_makespan(time_costs: ndarray,
                 assignment: ndarray) -> float:
    """
    Computes the makespan of a given assignment of tasks to num_resources.

    Parameters
    ----------
    time_costs : np.ndarray(shape=(num_resources, tasks+1))
        Cost functions per resource (C)
    assignment : np.array(shape=(num_resources))
        Assignment of tasks to num_resources

    Returns
    -------
    numpy.float64
        Makespan

    Notes
    -----
    The makespan is the maximum cost among all num_resources based on the
    number of tasks assigned to them.
    """
    # gets the cost for each resource indexed by the assignment array
    cost_by_resource = time_costs[arange(len(time_costs)), assignment]
    # gets the maximum cost
    makespan = max(cost_by_resource)
    return makespan


def get_total_cost(costs: ndarray,
                   assignment: ndarray) -> float:
    """
    Computes the total cost of a given assignment of tasks to resources.

    Parameters
    ----------
    costs : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    assignment : np.array(shape=(resources))
        Assignment of tasks to resources

    Returns
    -------
    numpy.float64
        Total cost

    Notes
    -----
    The total cost is the sum of the cost for all resources based on the
    number of tasks assigned to them.
    """
    # gets the cost for each resource indexed by the assignment array
    cost_by_resource = costs[arange(len(costs)), assignment]
    # gets the total cost
    total_cost = sum(cost_by_resource)
    return total_cost


def check_limits(assignment: ndarray,
                 lower_limit: ndarray,
                 upper_limit: ndarray) -> bool:
    """
    Checks if any resource has less or more tasks than its limits.

    Parameters
    ----------
    assignment : np.array(shape=(num_resources))
        Assignment of tasks to num_resources
    lower_limit : np.array(shape=(num_resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(num_resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    boolean
        True is the limits are respected
    """
    # np.any returns True if any of the values is True
    # True is in this case happens if any assignments disrespect the limits
    # beware that np.any gives other values for normal arrays (not np.array)
    any_below = any(assignment < lower_limit)
    any_above = any(assignment > upper_limit)
    return not (any_below or any_above)


def check_total_assigned(tasks: int,
                         assignment: ndarray) -> bool:
    """
    Checks if the total number of tasks assigned matches the number of tasks.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    assignment : np.array(shape=(num_resources))
        Assignment of tasks to num_resources

    Returns
    -------
    boolean
        True is they match
    """
    return tasks == sum(assignment)
