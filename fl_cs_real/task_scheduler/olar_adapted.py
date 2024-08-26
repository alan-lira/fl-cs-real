from heapq import heapify, heappop, heappush
from numpy import array, copy, ndarray, sum


def olar_adapted(tasks: int,
                 resources: int,
                 cost: ndarray,
                 assignment_capacities: ndarray) -> ndarray:
    # Some remarks about this adapted version of OLAR algorithm:
    # 1. It receives one set of possible task assignments per resource instead of two sets (original algorithm):
    #    one with lower and one with upper task assignment limits.
    """
    Finds an assignment of tasks to resources using OLAR.

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
    heap = []
    lower_limits = array([assignment_capacity_i[0] for assignment_capacity_i in assignment_capacities])
    upper_limits = array([assignment_capacity_i[-1] for assignment_capacity_i in assignment_capacities])
    # Assigns lower limit to all resources
    assignment = copy(lower_limits)
    for i in range(resources):
        # Initializes the heap
        if assignment[i] < upper_limits[i]:
            i_index = list(assignment_capacities[i]).index(assignment[i])
            heap.append((cost[i][i_index], i))
    heapify(heap)
    # Computes zeta (sum of lower limits)
    zeta = sum(lower_limits)
    # Iterates assigning the remaining tasks
    for t in range(zeta+1, tasks+1):
        c, j = heappop(heap)  # Find minimum cost
        assignment[j] += 1  # Assigns task t
        # Checks if more tasks can be assigned to j
        if assignment[j] in assignment_capacities[j]:
            if assignment[j] < upper_limits[j]:
                j_index = list(assignment_capacities[j]).index(assignment[j])
                heappush(heap, (cost[j][j_index], j))
        else:
            heappush(heap, (c, j))
    return assignment
