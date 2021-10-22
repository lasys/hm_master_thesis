import mitiq
from .mitiq_cdr import execute_with_cdr
from mitiq.zne import mitigate_executor
from functools import partial
import cirq.circuits.circuit

from typing import Optional, List, Callable, Union, Dict, Tuple
import logging
import warnings
from time import time
import numpy as np

from qiskit import Aer
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

import sys
sys.path.append("..")
sys.path.append("../..")
from ..gem.gem_postprocess import format_counts

logger = logging.getLogger(__name__)

class CDR_VQE(VQE):
     
    def sample_bitstrings_simulator(self, circuit, shots: int = 1024) -> Dict[int, int]:
        # Use Aer's qasm_simulator
        print("run simulator")
        if type(circuit) == cirq.circuits.circuit.Circuit:
            circuit = mitiq.interface.mitiq_qiskit.conversions.to_qiskit(circuit)
            circuit.measure_all()
        backend_sim = Aer.get_backend('qasm_simulator')
        qi = QuantumInstance(backend=backend_sim, shots=self._quantum_instance._run_config.shots)
        result = qi.execute(circuit)

        counts = result.get_counts()
        formatted_counts = format_counts(counts, result.results[0].header)
        int_counts = {}
        for key in formatted_counts.keys():
            int_counts[int(key.replace(' ',''),2)] = formatted_counts[key]
            
        return int_counts
    
    
    def sample_bitstrings(self, circuits, shots: int = 1024) -> Dict[int, int]:
        print("run noise")
        
        qiskit_circuits = []
        for circuit in circuits:
        
            # convert cirq.circuit to qiskit-circuit 
            if type(circuit) == cirq.circuits.circuit.Circuit:
                circuit = mitiq.interface.mitiq_qiskit.conversions.to_qiskit(circuit)
            circuit.measure_all()
            qiskit_circuits.append(circuit)
        
        result = self._quantum_instance.execute(qiskit_circuits)
        circuit_int_counts = []
        
        for i in range(0, len(qiskit_circuits)):
            
            counts = result.results[i].data.counts.copy()
            formatted_counts = format_counts(counts, result.results[i].header)
            int_counts = {}
            for key in formatted_counts.keys():
                int_counts[int(key.replace(' ',''),2)] = formatted_counts[key]

            circuit_int_counts.append(int_counts.copy())
          
        return circuit_int_counts
    
    
    def get_energy_evaluation(
        self,
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Returns a function handle to evaluates the energy at given parameters for the ansatz.

        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            operator: The operator whose energy to evaluate.
            return_expectation: If True, return the ``ExpectationBase`` expectation converter used
                in the construction of the expectation value. Useful e.g. to evaluate other
                operators with the same expectation value converter.


        Returns:
            Energy of the hamiltonian of each parameter, and, optionally, the expectation
            converter.

        Raises:
            RuntimeError: If the circuit is not parameterized (i.e. has 0 free parameters).

        """
        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        expect_op, expectation = self.construct_expectation(
            self._ansatz_params, operator, return_expectation=True
        )

        # create circuit for CDR 
        param_dict = dict(zip(self._ansatz_params, self._initial_point))  # type: Dict
        circuit = self.ansatz.assign_parameters(param_dict)
      
        def energy_evaluation(parameters):
     
            if circuit != None:
                
                # Observable(s) to measure.
                obs = np.diag(operator.to_matrix())
                
                means = execute_with_cdr(
                    circuit=circuit,
                    executor=self.sample_bitstrings,
                    observables=[obs],
                    simulator=self.sample_bitstrings_simulator,
                )

                return means if len(means) > 1 else means[0]
            
            print("Don't use cdr")
            return energy_evaluation_2(None, parameters)


        def energy_evaluation_2(cc, parameters):
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            # Create dict associating each parameter with the lists of parameterization values for it
            param_bindings = dict(zip(self._ansatz_params, parameter_sets.transpose().tolist()))       
            start_time = time()
            sampled_expect_op = self._circuit_sampler.convert(expect_op, params=param_bindings)
            means = np.real(sampled_expect_op.eval())
            if self._callback is not None:
                variance = np.real(expectation.compute_variance(sampled_expect_op))
                estimator_error = np.sqrt(variance / self.quantum_instance.run_config.shots)
                for i, param_set in enumerate(parameter_sets):
                    self._eval_count += 1
                    self._callback(self._eval_count, param_set, means[i], estimator_error[i])
            else:
                self._eval_count += len(means)

            end_time = time()
            logger.info(
                "Energy evaluation returned %s - %.5f (ms), eval count: %s",
                means,
                (end_time - start_time) * 1000,
                self._eval_count,
            )

            return means if len(means) > 1 else means[0]

        if return_expectation:
            return energy_evaluation, expectation

        return energy_evaluation
    