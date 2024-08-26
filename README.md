# Client Selection Algorithms for Federated Learning (Real-World Experiments)

Set of algorithms that select clients in Federated Learning (FL) systems with heterogeneous resources.

The algorithms can be evaluated via **real-world experiments** under the Flower [5] framework.

### Implemented Algorithms

#### Our Contributions:

- *Minimal Makespan and Energy Consumption FL Schedule* — **MEC**: finds an **optimal** solution that minimizes training time and total energy consumption, in that order, by defining how much data each client should use;


- *Minimal Energy Consumption and Makespan FL Schedule Under Time Constraint* — **ECMTC**: finds an **optimal** solution that minimizes total energy consumption and training time, in that order, while also meeting a deadline, by defining how much data each client should use.

#### Related Contributions:

- *Energy and Latency-aware Resource Management and Client Selection* — **ELASTIC** [1]: selects clients by dynamically adjusting the trade-off between maximizing the number of selected clients and minimizing the total energy consumption;


- *Federated Learning for Accuracy-Energy-Based Client Selection* — **FedAECS** [2]: optimizes the trade-off between energy consumption and accuracy to handle clients with dissimilar amounts of data;


- *OptimaL Assignment of tasks to Resources* — **OLAR** [3]: finds an **optimal** solution that minimizes the duration of a communication round by greedily controlling how much data each client uses;


- *Multiple-Choice Minimum-Cost Maximal Knapsack Packing Problem* — **(MC)²MKP** [4]: finds an **optimal** solution, based on the multiple-choice minimum-cost maximal knapsack packing problem, that minimizes energy consumption while controlling the workload distribution on heterogeneous resources.


### References

[1] Yu, L., Albelaihi, R., Sun, X., et al.: Jointly Optimizing Client Selection and Resource Management in Wireless Federated Learning for Internet of Things. IEEE Internet of Things Journal 9(6), 4385–4395 (2021)

[2] Zheng, J., Li, K., Tovar, E., Guizani, M.: Federated Learning for Energy-balanced Client Selection in Mobile Edge Computing. In: International Wireless Communications and Mobile Computing. pp. 1942–1947 (2021)

[3] Pilla, L.L.: Optimal Task Assignment for Heterogeneous Federated Learning Devices. In: IEEE International Parallel and Distributed Processing Symposium. pp. 661–670 (2021)

[4] Pilla, L.L.: Scheduling Algorithms for Federated Learning With Minimal Energy Consumption. IEEE Transactions on Parallel and Distributed Systems 34(4), 1215–1226 (2023)

[5] Beutel, D.J., Topal, T., Mathur, A., Qiu, X., Parcollet, T., de Gusmão, P.P., Lane, N.D.: Flower: A Friendly Federated Learning Research Framework. arXiv (2020)
