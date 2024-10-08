from numpy import full, inf, ndarray, zeros


def mc2mkp_adapted(tasks: int,
                   resources: int,
                   cost: ndarray,
                   assignment_capacities: ndarray) -> ndarray:
    # Some remarks about this adapted version of (MC)2MKP algorithm:
    # 1. It receives one set of possible task assignments per resource instead of two sets (original algorithm):
    #    one with lower and one with upper task assignment limits.
    """
    Finds an assignment of tasks to resources based on the dynamic
    programming algorithm for the (MC)^2MKP problem.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    assignment_capacities : ndarray(shape=(resources))
        Task assignment capacities per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # Initialization
    # K = minimal costs
    # I = Partial solutions (schedule for a given resource and t)
    K = full(shape=(resources, tasks+1), fill_value=inf)
    I = zeros(shape=(resources, tasks+1), dtype=int)
    # Solutions for Z_1
    for j in assignment_capacities[0]:
        if j <= tasks:
            j_index = list(assignment_capacities[0]).index(j)
            K[0][j] = cost[0][j_index]
            I[0][j] = j
    # Solutions for Z_i
    for i in range(1, resources):
        # All possible values for x_i
        for j in assignment_capacities[i]:
            j_index = list(assignment_capacities[i]).index(j)
            c = cost[i][j_index]
            for t in range(j, tasks+1):
                if K[i-1][t-j] + c < K[i][t]:
                    # New best solution for Z_i(t)
                    K[i][t] = K[i-1][t-j] + c
                    I[i][t] = j
    # Gets the final assignment from the support matrices
    assignment = zeros(resources, dtype=int)
    t = tasks
    for i in reversed(range(resources)):
        j = I[i][t]  # Number of tasks to resource i
        assignment[i] = j
        t = t-j      # index for the solution for resource i-1
    return assignment
