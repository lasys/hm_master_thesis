# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" The Quantum Approximate Optimization Algorithm with ZNE. """

import logging
from typing import Dict

from qiskit.opflow import (
    StateFn,
)

logger = logging.getLogger(__name__)

from typing import List, Union

from qiskit.circuit import QuantumCircuit

from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOA

class WS_QAOA(QAOA):
    
        
    def _get_eigenstate(self, optimal_parameters) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the ansatz, provided with parameters."""

        # Quickfix!
        wave_function = self.ansatz.bind_parameters(optimal_parameters)
        temp = []
        counter = 0
        qasm = wave_function.qasm()
        qasm_new = ""
        for line in qasm.split('\n'):
            if 'rz(-pi)' in line:
                counter += 1
                if line in temp:
                    line = line.replace('-pi','0.0')
                else:
                    temp.append(line)
            qasm_new += line + "\n"
        
        wave_function = QuantumCircuit.from_qasm_str(qasm_new)
        state_fn = self._circuit_sampler.convert(StateFn(wave_function)).eval()
        if self.quantum_instance.is_statevector:
            state = state_fn.primitive.data  # VectorStateFn -> Statevector -> np.array
        else:
            state = state_fn.to_dict_fn().primitive  # SparseVectorStateFn -> DictStateFn -> dict

        return state
    
    