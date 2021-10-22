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

""" The Quantum Approximate Optimization Algorithm with No Optimizer. """

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

class No_Opt_QAOA(QAOA):
       
    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> MinimumEigensolverResult:
        
        #super().compute_minimum_eigenvalue(operator, aux_operators)

        if self.quantum_instance is None:
            raise AlgorithmError(
                "A QuantumInstance or Backend must be supplied to run the quantum algorithm."
            )
        self.quantum_instance.circuit_summary = True

        # this sets the size of the ansatz, so it must be called before the initial point
        # validation
        self._check_operator_ansatz(operator)

        # set an expectation for this algorithm run (will be reset to None at the end)
        initial_point = _validate_initial_point(self.initial_point, self.ansatz)

        bounds = _validate_bounds(self.ansatz)
        
        # We need to handle the array entries being Optional i.e. having value None
        if aux_operators:
            zero_op = I.tensorpower(operator.num_qubits) * 0.0
            converted = []
            for op in aux_operators:
                if op is None:
                    converted.append(zero_op)
                else:
                    converted.append(op)

            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes.
            aux_operators = [zero_op if op == 0 else op for op in converted]
        else:
            aux_operators = None 
            
        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        if isinstance(self._gradient, GradientBase):
            gradient = self._gradient.gradient_wrapper(
                ~StateFn(operator) @ StateFn(self._ansatz),
                bind_params=self._ansatz_params,
                backend=self._quantum_instance,
            )
        else:
            gradient = self._gradient

        #self._eval_count = 0
        #energy_evaluation, expectation = self.get_energy_evaluation(
        #    operator, return_expectation=True
        #)
        
        result = VQEResult()
        result.optimal_point = 0.0
        result.optimal_parameters = dict()
        #result.optimal_value = opt_result.fun
        #result.cost_function_evals = opt_result.nfev
        #result.optimizer_time = eval_time
        #result.eigenvalue = opt_result.fun + 0j
        result.eigenstate = self._get_eigenstate(initial_point)

        #logger.info(
        #    "Optimization complete in %s seconds.\nFound opt_params %s in %s evals",
        #    eval_time,
        #    result.optimal_point,
        #    self._eval_count,
        #)

        # TODO delete as soon as get_optimal_vector etc are removed
        self._ret = result

        if aux_operators is not None:
            aux_values = self._eval_aux_ops(opt_result.x, aux_operators, expectation=expectation)
            result.aux_operator_eigenvalues = aux_values[0]

        return result
