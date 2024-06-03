from math import inf
from numpy import ndarray, ones


def calculate_ni_adapted(α: float,
                         Ei: float) -> float:
    return α * (Ei + 1) - 1


def elastic_adapted_client_selection_algorithm(I: int,
                                               A: ndarray,
                                               t: ndarray,
                                               E: ndarray,
                                               τ: float,
                                               α: float) -> tuple:
    # Some remarks about this adapted version of ELASTIC algorithm:
    # 1. We considered that clients do not share a wireless channel, so they can upload their model
    #    without having to wait for the channel availability. In other words, ∀i ∈ I, t_wait_i = 0.
    # 2. The algorithm receives a previously generated array of task assignment capacities (A),
    #    such that the i-th client can process exactly Ai tasks.
    # 3. The algorithm receives a previously generated array of time costs (t), such that ti = t_comp_i + t_up_i.
    # 4. The algorithm receives a previously generated array of energy costs (E), such that Ei = E_comp_i + E_up_i.

    # ∀i ∈ I, compute ηi.
    idx = []
    n = []
    for i in range(I):
        idx.append(i)
        if A[i] == 0:
            ni = inf
        else:
            ni = calculate_ni_adapted(α, float(E[i][A[i]]))
        n.append(ni)
    # Sort all the clients in increasing order based on ηi.
    # Denote I′ as the set of sorted clients.
    sorted_n, sorted_idx = map(list, zip(*sorted(zip(n, idx), reverse=False)))
    # Initialize x.
    x = ones(shape=(len(sorted_idx)), dtype=int)
    for _ in enumerate(sorted_idx):
        # Update the set of participants J based on Constraints (13) and (14).
        # Constraints (13) and (14) define the set of selected clients, which are sorted based on the
        # increasing order of their computational latency.
        # Constraint 13: J = {i ∈ I| xi = 1}
        # Constraint 14: ∀j ∈ J, t_comp_j ≤ t_comp_j+1 (Considering this adaptation, t_j ≤ t_j+1)
        idx_j = []
        t_j = []
        for index, _ in enumerate(x):
            if x[index] == 1:
                idxj = idx[index]
                idx_j.append(idxj)
                tj = t[index][A[index]-1]
                t_j.append(tj)
        sorted_t_j, sorted_J = map(list, zip(*sorted(zip(t_j, idx_j), reverse=False)))
        for index, _ in enumerate(sorted_J):
            idxj = idx_j[index]
            if sorted_t_j[index] > τ or A[idxj] == 0:
                x[idxj] = 0
                break
    # Organize the solution.
    tasks_assignment = []
    selected_clients = []
    makespan = 0
    energy_consumption = 0
    for index, _ in enumerate(x):
        tasks_assignment.append(0)
        if x[index] == 1:
            j = idx[index]  # Display the selected clients in ascending order.
            # j = sorted_idx[index]  # Display the selected clients sorted by n.
            j_num_tasks_assigned = A[j]
            tasks_assignment[index] = j_num_tasks_assigned
            selected_clients.append(j)
            makespan_j = t[j][j_num_tasks_assigned]
            if makespan_j > makespan:
                makespan = makespan_j
            energy_consumption_j = E[j][j_num_tasks_assigned]
            energy_consumption += energy_consumption_j
    return x, tasks_assignment, selected_clients, makespan, energy_consumption
