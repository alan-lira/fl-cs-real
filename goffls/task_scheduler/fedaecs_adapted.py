from itertools import combinations
from math import inf, log
from numpy import argmin, array, ndarray, zeros


def calculate_Γ(sum_εi: float) -> float:
    return log(1 + sum_εi)


def fedaecs_adapted(I: int,
                    K: int,
                    num_tasks: int,
                    A: ndarray,
                    T: ndarray,
                    E: ndarray,
                    ε: ndarray,
                    b: ndarray,
                    ε0: float,
                    T_max: float,
                    B: float) -> tuple:
    # Some remarks about this adapted version of FedAECS algorithm:
    # 1. We considered that clients do not share a channel with limited bandwidth (B = ∞),
    #    so it doesn't matter their bandwidth information. In other words, ∀i ∈ I ∀k ∈ K, bik = 0.
    # 2. The algorithm receives the total number of tasks to be allocated among the selected clients (num_tasks),
    #    such that the algorithm won't stop until this constraint is also respected.
    # 3. The algorithm receives a previously generated array of task assignment capacities (A), ∀i ∈ I ∀k ∈ K,
    #    such that at the ith round the client k can process between Aik_min (inclusive) and Aik_max (inclusive) tasks.
    #    Note that Aik specifies the number of tasks while Dik originally specified the size of tasks.
    # 4. The algorithm receives a previously generated array of time costs (T),
    #    such that Tika = t_train_ika + t_up_ika, ∀i ∈ I ∀k ∈ K ∀a ∈ A.
    # 5. The algorithm receives a previously generated array of energy costs (E),
    #    such that Eika = E_comp_ika + E_up_ika, ∀i ∈ I ∀k ∈ K ∀a ∈ A.
    # 6. The algorithm receives a previously generated array of training accuracies (ε), ∀i ∈ I ∀k ∈ K ∀a ∈ A.
    beta_star = []
    beta_star_tasks = []
    f_obj_beta_star = []
    selected_clients = []
    makespan = []
    energy_consumption = []
    training_accuracy = []
    # For each communication round i.
    for i in range(I):
        # Get the training accuracy of clients.
        εi = ε[i]
        # Get the total energy consumption (E_comp_ik + E_up_ik) of clients.
        Ei = E[i]
        # Get the total time (t_train_ik + t_up_ik) of clients.
        Ti = T[i]
        # Auxiliary variables.
        n = []
        init_qualified_client_capacity = []
        init_qualified_client_max_capacity = []
        init_qualified_client_energy = []
        init_qualified_client_accuracy = []
        init_qualified_client_time = []
        init_qualified_client_bandwidth = []
        init_qualified_client_index = []
        init_unqualified_client_index = []
        # Optimization variables.
        beta_line_i = zeros(shape=K, dtype=int)
        beta_star_i = zeros(shape=K, dtype=int)
        beta_star_tasks_i = zeros(shape=K, dtype=int)
        f_obj_beta_star_i = inf
        # Check the bandwidth and time constraints (preliminary screening).
        for k in range(K):
            Ak = A[i][k]
            Ak_max = max(Ak)
            init_qualified_client_max_capacity.append(Ak_max)
            for index, _ in enumerate(Ak):
                Ak_index = list(Ak).index(Ak[index])
                if Ti[k][Ak_index] <= T_max and b[i][k][Ak_index] <= B:
                    if εi[k][Ak_index] > 0:
                        n.append(Ei[k][Ak_index] / εi[k][Ak_index])
                    else:
                        n.append(inf)
                    init_qualified_client_capacity.append(Ak[Ak_index])
                    init_qualified_client_energy.append(Ei[k][Ak_index])
                    init_qualified_client_accuracy.append(εi[k][Ak_index])
                    init_qualified_client_time.append(Ti[k][Ak_index])
                    init_qualified_client_bandwidth.append(b[i][k][Ak_index])
                    init_qualified_client_index.append(k)
                else:
                    init_unqualified_client_index.append(k)
        # Output the unqualified clients.
        if init_unqualified_client_index:
            for unqualified_client_index in init_unqualified_client_index:
                beta_line_i[unqualified_client_index] = 1
        # Sort n in ascending order (according to the ratio of the energy consumption and the FL accuracy).
        sorted_client_n = []
        sorted_client_capacity = []
        sorted_client_energy = []
        sorted_client_accuracy = []
        sorted_client_bandwidth = []
        sorted_client_index = []
        if init_qualified_client_index:
            (sorted_client_n,
             sorted_client_capacity,
             sorted_client_energy,
             sorted_client_accuracy,
             sorted_client_bandwidth,
             sorted_client_index) \
                = map(list, zip(*sorted(zip(n,
                                            init_qualified_client_capacity,
                                            init_qualified_client_energy,
                                            init_qualified_client_accuracy,
                                            init_qualified_client_bandwidth,
                                            init_qualified_client_index),
                                        reverse=False)))
        # Initializing m.
        m = 0
        num_tasks_assigned = 0
        found_solution = False
        while m <= len(sorted_client_n) - 1:
            if sorted_client_accuracy[m] >= ε0:
                client_index = sorted_client_index[m]
                if (beta_star_tasks_i[client_index] + sorted_client_capacity[m]
                        <= init_qualified_client_max_capacity[client_index]):
                    tasks_to_assign = sorted_client_capacity[m]
                else:
                    tasks_to_assign = init_qualified_client_max_capacity[client_index] - beta_star_tasks_i[client_index]
                if num_tasks_assigned + tasks_to_assign > num_tasks:
                    tasks_to_assign = num_tasks - num_tasks_assigned
                beta_star_i[client_index] = 1
                beta_star_tasks_i[client_index] += tasks_to_assign
                num_tasks_assigned += tasks_to_assign
                f_obj_beta_star_i = sorted_client_n[m]
                if num_tasks_assigned == num_tasks:
                    found_solution = True
            else:
                if sorted_client_accuracy[m] >= ε0:
                    client_index = sorted_client_index[m]
                    if beta_star_tasks_i[client_index] + sorted_client_capacity[m] \
                            <= init_qualified_client_max_capacity[client_index]:
                        tasks_to_assign = sorted_client_capacity[m]
                    else:
                        tasks_to_assign \
                            = init_qualified_client_max_capacity[client_index] - beta_star_tasks_i[client_index]
                    if num_tasks_assigned + tasks_to_assign > num_tasks:
                        tasks_to_assign = num_tasks - num_tasks_assigned
                    beta_star_i[client_index] = 1
                    beta_star_tasks_i[client_index] += tasks_to_assign
                    num_tasks_assigned += tasks_to_assign
                    the_first_qualified_client_index = beta_star_i
                    the_first_qualified_client_index_tasks = beta_star_tasks_i
                    # Check the combination selection of the previous clients.
                    selection_possibilities = []
                    for mi in range(0, m):
                        s = array(list(combinations(range(0, m), mi)))
                        si_to_remove = []
                        for si_idx, si in enumerate(s):
                            client_indices = []
                            for mi_idx in si:
                                client_index = sorted_client_index[mi_idx]
                                if client_index not in client_indices:
                                    client_indices.append(client_index)
                                else:
                                    si_to_remove.append(si_idx)
                        si_to_remove = set(si_to_remove)
                        s = [i for j, i in enumerate(s) if j not in si_to_remove]
                        s_combinations = [list(si) for si in s]
                        for s_combination in s_combinations:
                            if s_combination and s_combination not in selection_possibilities:
                                capacity_selected = sum([sorted_client_capacity[mi] for mi in s_combination])
                                if capacity_selected + num_tasks_assigned >= num_tasks:
                                    selection_possibilities.append(s_combination)
                    qualified_selection = []
                    obj = []
                    # Calculate the model accuracy and total bandwidth for each selection possibility.
                    for selection_possibility in selection_possibilities:
                        sum_accuracy_select_idx = 0
                        sum_bandwidth_select_idx = 0
                        sum_tasks_select_idx = 0
                        for idx in selection_possibility:
                            sum_accuracy_select_idx += sorted_client_accuracy[idx]
                            sum_bandwidth_select_idx += sorted_client_bandwidth[idx]
                            sum_tasks_select_idx += sorted_client_capacity[idx]
                        model_accuracy_select_idx = calculate_Γ(sum_accuracy_select_idx)
                        # Check the constraints are whether satisfied.
                        if (model_accuracy_select_idx >= ε0 and
                                sum_bandwidth_select_idx <= B and
                                sum_tasks_select_idx + num_tasks_assigned >= num_tasks):
                            # Calculate the total energy consumption of the qualified selection.
                            total_energy_qualified_select_idx = 0
                            for idx in selection_possibility:
                                total_energy_qualified_select_idx += sorted_client_energy[idx]
                            # Calculate the objective function.
                            if model_accuracy_select_idx > 0:
                                f_obj = total_energy_qualified_select_idx / model_accuracy_select_idx
                            else:
                                f_obj = inf
                            obj = list(obj)
                            # Store the objective function value.
                            obj.append(f_obj)
                            # Store the qualified selection.
                            qualified_selection.append(selection_possibility)
                    obj = array(obj)
                    # Check whether there is a client selection for combinatorial optimization
                    # satisfying constraints.
                    if qualified_selection:
                        # y is the location (index) of objective function minimum value.
                        y = argmin(obj)
                        # Further compare the optimal values for the objective function.
                        if obj[y] <= sorted_client_n[m]:
                            f_obj_beta_star_i = obj[y]
                            for qs_idx in qualified_selection[y]:
                                client_index = sorted_client_index[qs_idx]
                                if beta_star_tasks_i[client_index] + sorted_client_capacity[qs_idx] \
                                        <= init_qualified_client_max_capacity[client_index]:
                                    tasks_to_assign = sorted_client_capacity[qs_idx]
                                else:
                                    tasks_to_assign = (init_qualified_client_max_capacity[client_index]
                                                       - beta_star_tasks_i[client_index])
                                if num_tasks_assigned + tasks_to_assign > num_tasks:
                                    tasks_to_assign = num_tasks - num_tasks_assigned
                                beta_star_i[client_index] = 1
                                beta_star_tasks_i[client_index] += tasks_to_assign
                                num_tasks_assigned += tasks_to_assign
                        else:
                            f_obj_beta_star_i = sorted_client_n[m]
                            beta_star_i = the_first_qualified_client_index
                            beta_star_tasks_i = the_first_qualified_client_index_tasks
                        if num_tasks_assigned == num_tasks:
                            found_solution = True
                    else:
                        beta_star_i = the_first_qualified_client_index
                        beta_star_tasks_i = the_first_qualified_client_index_tasks
            if found_solution:
                break
            m = m + 1
        # Organizing the solution for the round i.
        selected_clients_i = []
        makespan_i = 0
        energy_consumption_i = 0
        sum_accuracy_i = 0
        for client_index, _ in enumerate(beta_star_i):
            if beta_star_i[client_index] == 1:
                selected_clients_i.append(client_index)
                selected_client_num_tasks = beta_star_tasks_i[client_index]
                selected_client_num_tasks_index = list(A[i][client_index]).index(selected_client_num_tasks)
                makespan_ik = Ti[client_index][selected_client_num_tasks_index]
                if makespan_ik > makespan_i:
                    makespan_i = makespan_ik
                energy_consumption_ik = Ei[client_index][selected_client_num_tasks_index]
                energy_consumption_i += energy_consumption_ik
                accuracy_ik = εi[client_index][selected_client_num_tasks_index]
                sum_accuracy_i += accuracy_ik
        beta_star.append(beta_star_i)
        beta_star_tasks.append(beta_star_tasks_i)
        f_obj_beta_star.append(f_obj_beta_star_i)
        selected_clients.append(selected_clients_i)
        makespan.append(makespan_i)
        energy_consumption.append(energy_consumption_i)
        accuracy_i = calculate_Γ(sum_accuracy_i)
        training_accuracy.append(accuracy_i)
    return (beta_star, beta_star_tasks, f_obj_beta_star, selected_clients, makespan, energy_consumption,
            training_accuracy)
