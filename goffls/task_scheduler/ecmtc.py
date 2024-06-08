from numpy import full, inf, ndarray, zeros


def ecmtc(num_resources: int,
          num_tasks: int,
          assignment_capacities: ndarray,
          time_costs: ndarray,
          energy_costs: ndarray,
          time_limit: float) -> tuple:
    """
    Minimal Energy Consumption and Makespan FL Schedule under Time Constraint (ECMTC): finds an optimal schedule (X*)
    that minimizes the total energy consumption (ΣE) and the makespan (C_max), in that order,
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
    optimal_schedule : ndarray(shape=(num_resources), int), minimal_energy_consumption : float, minimal_makespan : float
        Optimal schedule (X*), minimal energy consumption (ΣE), and minimal makespan (C_max)
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
    minimal_time_costs = full(shape=(num_resources, num_tasks+1), fill_value=inf, dtype=float)
    # (III) Solutions for the first resource (Z₁).
    for j in assignment_capacities_filtered[0]:
        j_index = list(assignment_capacities[0]).index(j)
        partial_solutions[0][j] = j
        minimal_energy_costs[0][j] = energy_costs[0][j_index]
        minimal_time_costs[0][j] = time_costs[0][j_index]
    # Solutions for other resources (Zᵢ).
    for i in range(1, num_resources):
        # Test all assignments to resource i.
        for j in assignment_capacities_filtered[i]:
            j_index = list(assignment_capacities[i]).index(j)
            for t in range(j, num_tasks+1):
                # (IV) Test new solution.
                energy_cost_new_solution = minimal_energy_costs[i-1][t-j] + energy_costs[i][j_index]
                time_cost_new_solution = max(float(minimal_time_costs[i-1][t-j]), float(time_costs[i][j_index]))
                if ((energy_cost_new_solution < minimal_energy_costs[i][t]) or
                        (energy_cost_new_solution == minimal_energy_costs[i][t] and
                         time_cost_new_solution < minimal_time_costs[i][t])):
                    # New best solution for Zᵢ(t).
                    minimal_energy_costs[i][t] = energy_cost_new_solution
                    minimal_time_costs[i][t] = time_cost_new_solution
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
    minimal_makespan = minimal_time_costs[num_resources-1][num_tasks]
    # Return the optimal schedule (X*), the minimal energy consumption (ΣE), and the minimal makespan (C_max).
    return optimal_schedule, minimal_energy_consumption, minimal_makespan
