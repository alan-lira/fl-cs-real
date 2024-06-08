from numpy import full, inf, ndarray, zeros


def min_max_time(num_resources: int,
                 num_tasks: int,
                 assignment_capacities: ndarray,
                 time_costs: ndarray) -> float:
    """
    First step of MEC: finds the minimal makespan (C_max) to assign tasks to resources.
    Parameters
    ----------
    num_resources : int
        Number of resources (n)
    num_tasks : int
        Number of tasks (T)
    assignment_capacities : ndarray(shape=(num_resources), int)
        Task assignment capacities per resource (A)
    time_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Time costs to process tasks per resource (Ρ)
    Returns
    -------
    minimal_makespan : float
        Minimal makespan (C_max)
    """
    # (I) Filtering: nothing to do.
    # (II) Initialization: minimal costs and partial solutions matrices.
    partial_solutions = zeros(shape=(num_resources, num_tasks+1), dtype=int)
    minimal_time_costs = full(shape=(num_resources, num_tasks+1), fill_value=inf, dtype=float)
    # (III) Solutions for the first resource (Z₁).
    for j in assignment_capacities[0]:
        j_index = list(assignment_capacities[0]).index(j)
        partial_solutions[0][j] = j
        minimal_time_costs[0][j] = time_costs[0][j_index]
    # Solutions for other resources (Zᵢ).
    for i in range(1, num_resources):
        # Test all assignments to resource i.
        for j in assignment_capacities[i]:
            j_index = list(assignment_capacities[i]).index(j)
            for t in range(j, num_tasks+1):
                # (IV) Test new solution.
                time_cost_new_solution = max(float(minimal_time_costs[i-1][t-j_index]), float(time_costs[i][j_index]))
                if time_cost_new_solution < minimal_time_costs[i][t]:
                    # New best solution for Zᵢ(t).
                    minimal_time_costs[i][t] = time_cost_new_solution
                    partial_solutions[i][t] = j
    # (V) Organize the final solution.
    minimal_makespan = float(minimal_time_costs[num_resources-1][num_tasks])
    # Return the minimal makespan (C_max).
    return minimal_makespan


def min_sum_energy(num_resources: int,
                   num_tasks: int,
                   assignment_capacities: ndarray,
                   time_costs: ndarray,
                   energy_costs: ndarray,
                   time_limit: float) -> tuple:
    """
    Second step of MEC: finds the minimal energy consumption (ΣE) to assign tasks to resources,
    while meeting a deadline (D).
    Parameters
    ----------
    num_resources : int
        Number of resources (n)
    num_tasks : int
        Number of tasks (T)
    assignment_capacities : ndarray(shape=(num_resources), int)
        Task assignment capacities per resource (A)
    time_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Time costs to process tasks per resource (Ρ)
    energy_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Energy costs to process tasks per resource (ε)
    time_limit : float
        Deadline (D)
    Returns
    -------
    optimal_schedule : ndarray(shape=(num_resources), int), minimal_energy_consumption : float
        Optimal schedule (X*) and minimal energy consumption (ΣE)
    """
    # (I) Filtering: only assignments that respect the time limit (C).
    assignment_capacities_filtered = []
    for i in range(0, num_resources):
        assignment_capacities_i = []
        for j in assignment_capacities[i]:
            j_index = list(assignment_capacities[i]).index(j)
            if time_costs[i][j_index] <= time_limit:
                assignment_capacities_i.append(j)
        assignment_capacities_filtered.append(assignment_capacities_i)
    # (II) Initialization: minimal costs and partial solutions matrices.
    partial_solutions = zeros(shape=(num_resources, num_tasks+1), dtype=int)
    minimal_energy_costs = full(shape=(num_resources, num_tasks+1), fill_value=inf, dtype=float)
    # (III) Solutions for the first resource (Z₁).
    for j in assignment_capacities_filtered[0]:
        j_index = list(assignment_capacities[0]).index(j)
        partial_solutions[0][j] = j
        minimal_energy_costs[0][j] = energy_costs[0][j_index]
    # Solutions for other resources (Zᵢ).
    for i in range(1, num_resources):
        # Test all assignments to resource i.
        for j in assignment_capacities_filtered[i]:
            j_index = list(assignment_capacities[i]).index(j)
            for t in range(j, num_tasks+1):
                # (IV) Test new solution.
                energy_cost_new_solution = minimal_energy_costs[i-1][t-j_index] + energy_costs[i][j_index]
                if energy_cost_new_solution < minimal_energy_costs[i][t]:
                    # New best solution for Zᵢ(t).
                    minimal_energy_costs[i][t] = energy_cost_new_solution
                    partial_solutions[i][t] = j
    # Extract the optimal schedule (X*).
    t = num_tasks
    optimal_schedule = zeros(num_resources, dtype=int)
    for i in reversed(range(num_resources)):
        j = partial_solutions[i][t]  # Number of tasks to assign to resource i.
        optimal_schedule[i] = j
        t = t-j  # Solution index of resource i-1.
    # (V) Organize the final solution.
    minimal_energy_consumption = minimal_energy_costs[num_resources-1][num_tasks]
    # Return the optimal schedule (X*) and the minimal energy consumption (ΣE).
    return optimal_schedule, minimal_energy_consumption


def mec(num_resources: int,
        num_tasks: int,
        assignment_capacities: ndarray,
        time_costs: ndarray,
        energy_costs: ndarray) -> tuple:
    """
    Minimal Makespan and Energy Consumption FL Schedule (MEC): finds an optimal schedule (X*) that
    minimizes the makespan (C_max) and the total energy consumption (ΣE), in that order.
    Parameters
    ----------
    num_resources : int
        Number of resources (n)
    num_tasks : int
        Number of tasks (T)
    assignment_capacities : ndarray(shape=(num_resources), int)
        Task assignment capacities per resource (A)
    time_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Time costs to process tasks per resource (Ρ)
    energy_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Energy costs to process tasks per resource (ε)
    Returns
    -------
    optimal_schedule : ndarray(shape=(num_resources), int), minimal_makespan : float, minimal_energy_consumption : float
        Optimal schedule (X*), minimal makespan (C_max), and minimal energy consumption (ΣE)
    """
    # First step: compute the MinMaxTime algorithm (minimal makespan optimization).
    minimal_makespan = min_max_time(num_resources,
                                    num_tasks,
                                    assignment_capacities,
                                    time_costs)
    # Second step: compute the MinSumEnergy algorithm (minimal energy consumption optimization).
    optimal_schedule, minimal_energy_consumption = min_sum_energy(num_resources,
                                                                  num_tasks,
                                                                  assignment_capacities,
                                                                  time_costs,
                                                                  energy_costs,
                                                                  minimal_makespan)
    # Return the optimal schedule (X*), the minimal makespan (C_max), and the minimal energy consumption (ΣE).
    return optimal_schedule, minimal_makespan, minimal_energy_consumption
