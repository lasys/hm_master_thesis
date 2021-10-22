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

from typing import Optional, List, Callable, Union, Dict, Tuple
import logging
import warnings
from time import time
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.opflow import (
    OperatorBase,
    ExpectationBase,
    ExpectationFactory,
    StateFn,
    CircuitStateFn,
    ListOp,
    I,
    CircuitSampler,
)
from qiskit.opflow.gradients import GradientBase
from qiskit.utils.validation import validate_min
from qiskit.utils.backend_utils import is_aer_provider
from qiskit.utils.deprecation import deprecate_function
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import Optimizer, SLSQP
from qiskit.algorithms.variational_algorithm import VariationalAlgorithm, VariationalResult
from qiskit.algorithms import MinimumEigensolver, MinimumEigensolverResult
from qiskit.algorithms.exceptions import AlgorithmError

from qiskit.algorithms.minimum_eigen_solvers.vqe import (
    VQE,
    VQEResult,
    _validate_initial_point,
    _validate_bounds
)

logger = logging.getLogger(__name__)

from typing import List, Callable, Optional, Union
import numpy as np

from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import OperatorBase, ExpectationBase
from qiskit.opflow.gradients import GradientBase
from qiskit.providers import Backend
from qiskit.providers import BaseBackend
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.utils.validation import validate_min
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz

from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOA
#from .no_opt_vqe import No_Opt_VQE

class WS_QAOA(QAOA):
    
        
    def _get_eigenstate(self, optimal_parameters) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the ansatz, provided with parameters."""
        
        #print(optimal_parameters)
        
        #for opti in optimal_parameters:
        #    if opti.name == "beta":
        #        optimal_parameters[opti] = np.pi/2
        #    else:
        #       optimal_parameters[opti] = 0.0
                
        
        
        
        wave_function = self.ansatz.bind_parameters(optimal_parameters)
        
         # print(f"my: {wave_function.qasm()}")
        
        
        temp = []
        
        counter = 0
        qasm = wave_function.qasm()
        #print(qasm)
        qasm_new = ""
        for line in qasm.split('\n'):
            if 'rz(-pi)' in line:
                counter += 1
                if line in temp:
                    line = line.replace('-pi','0.0')
                else:
                    temp.append(line)
                 
               # if counter > wave_function.num_qubits :
                    #print("changed")
               #     line = line.replace('-pi','46.0')
            qasm_new += line + "\n"
        
        #print(qasm_new)   
        #print(f"counter: {counter}")
        #print(wave_function.num_qubits)
        
        wave_function = QuantumCircuit.from_qasm_str(qasm_new)
        #print(wave_function)
        #wave_function.draw('mpl')
        state_fn = self._circuit_sampler.convert(StateFn(wave_function)).eval()
        if self.quantum_instance.is_statevector:
            state = state_fn.primitive.data  # VectorStateFn -> Statevector -> np.array
        else:
            state = state_fn.to_dict_fn().primitive  # SparseVectorStateFn -> DictStateFn -> dict

        return state
    
    