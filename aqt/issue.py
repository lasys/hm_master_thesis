
# https://github.com/Qiskit-Partners/qiskit-aqt-provider/issues/46

import networkx as nx
import numpy as np
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit_aqt_provider import AQTProvider
from qiskit_optimization.applications import Maxcut

# create graph
n = 4
graph = nx.Graph()
graph.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]
graph.add_weighted_edges_from(elist)

# create Maxcut
max_cut = Maxcut(graph)
max_cut_qubo = max_cut.to_quadratic_program()

# connection to AQT
aqt = AQTProvider('6f47670f3e5c414da0cdcab1c048eb97')
print(aqt.backends)
simulator_backend = aqt.backends.aqt_qasm_simulator
quantum_instance = QuantumInstance(backend=simulator_backend,shots=200)
qaoa = QAOA(optimizer=SPSA(maxiter=1), quantum_instance=quantum_instance)

# execute qaoa
MinimumEigenOptimizer(qaoa).solve(max_cut_qubo)

