from math import inf
from numpy import ndarray, ones


def calculate_ni_adapted(α: float,
                         Ei: float) -> float:
    return α * (Ei + 1) - 1


def elastic_adapted(I: int,
                    A: ndarray,
                    A_skd: ndarray,
                    t: ndarray,
                    E: ndarray,
                    τ: float,
                    α: float) -> tuple:
    # Some remarks about this adapted version of ELASTIC algorithm:
    # 1. We considered that clients do not share a wireless channel, so they can upload their model
    #    without having to wait for the channel availability. In other words, ∀i ∈ I, t_wait_i = 0.
    # 2. The algorithm receives a previously generated array of task assignment capacities (A),
    #    such that the i-th client can process a list of A_i tasks.
    # 3. The algorithm receives a previously generated array of tasks scheduled (A_skd),
    #    such that the i-th client will process exactly A_skd_i tasks.
    # 4. The algorithm receives a previously generated array of time costs (t), such that ti = t_comp_i + t_up_i.
    # 5. The algorithm receives a previously generated array of energy costs (E), such that Ei = E_comp_i + E_up_i.

    # ∀i ∈ I, compute ηi.
    idx = []
    n = []
    for i in range(I):
        idx.append(i)
        if A_skd[i] == 0:
            ni = inf
        else:
            A_skd_i_index = list(A[i]).index(A_skd[i])
            ni = calculate_ni_adapted(α, float(E[i][A_skd_i_index]))
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
                A_skd_i_index = list(A[index]).index(A_skd[index])
                tj = t[index][A_skd_i_index]
                t_j.append(tj)
        sorted_t_j, sorted_J = map(list, zip(*sorted(zip(t_j, idx_j), reverse=False)))
        for index, _ in enumerate(sorted_J):
            idxj = idx_j[index]
            if sorted_t_j[index] > τ or A_skd[idxj] == 0:
                x[idxj] = 0
                break
    # Organize the solution.
    tasks_assignment = []
    selected_clients = []
    for index, _ in enumerate(x):
        tasks_assignment.append(0)
        if x[index] == 1:
            j = idx[index]  # Display the selected clients in ascending order.
            # j = sorted_idx[index]  # Display the selected clients sorted by n.
            j_num_tasks_assigned = A_skd[j]
            tasks_assignment[index] = j_num_tasks_assigned
            selected_clients.append(j)
    return x, tasks_assignment, selected_clients
