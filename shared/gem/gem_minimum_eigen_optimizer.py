from qiskit.algorithms import MinimumEigensolver, MinimumEigensolverResult
from qiskit.utils import algorithm_globals
from qiskit.algorithms.minimum_eigen_solvers.vqe import _validate_initial_point, _validate_bounds
from qiskit_optimization.algorithms.minimum_eigen_optimizer import (
    MinimumEigenOptimizer,
    MinimumEigenOptimizationResult,
)
from qiskit.circuit import QuantumCircuit
from qiskit_optimization.converters.quadratic_program_converter import QuadraticProgramConverter
from qiskit_optimization.converters.quadratic_program_converter import QuadraticProgramConverter
from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.problems.variable import VarType
import copy
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Tuple, cast
from .gem_instance import GEMInstance
import numpy as np

class GEMMinimumEigenOptimizer(MinimumEigenOptimizer):
        
    def __init__(
        self,
        min_eigen_solver: MinimumEigensolver,
        gem: GEMInstance, 
        penalty: Optional[float] = None,
        converters: Optional[
            Union[QuadraticProgramConverter, List[QuadraticProgramConverter]]
        ] = None,
    ) -> None:
    
        self.gem = gem
        super().__init__(min_eigen_solver, penalty, converters)
    
    
    def _construct_ws_circuit(self):

        #print(self._min_eigen_solver._initial_point)
        wave_function = self._min_eigen_solver.ansatz.bind_parameters(self._min_eigen_solver._initial_point)
        #print(self._min_eigen_solver.ansatz)
        temp = []
        qasm = wave_function.qasm()
        qasm_new = ""
        for line in qasm.split('\n'):
            if 'rz(-pi)' in line:
                if line in temp:
                    line = line.replace('-pi','0.0')
                else:
                    temp.append(line)
            qasm_new += line + "\n"
       
        wave_function = QuantumCircuit.from_qasm_str(qasm_new)
        #print(wave_function)
        return wave_function 
        
        
    
    def _set_gem_matrix(self, operator, is_ws_circuit=False):
        
        # create random _initial_point if not exists 
        num_parameters = self._min_eigen_solver.ansatz.num_parameters
        if self._min_eigen_solver._initial_point is None:
            self._min_eigen_solver._initial_point = np.random.uniform(low=-2/np.pi, high=2/np.pi, size=(num_parameters,))
        
        # check and validate points and ansatz
        self._min_eigen_solver._check_operator_ansatz(operator)
        initial_point = _validate_initial_point(self._min_eigen_solver._initial_point, self._min_eigen_solver.ansatz)
        bounds = _validate_bounds(self._min_eigen_solver.ansatz)
        
        # construct circuit 
        param_dict = dict(zip(self._min_eigen_solver._ansatz_params, initial_point))  # type: Dict
        circuit = self._min_eigen_solver.ansatz.assign_parameters(param_dict)

        if is_ws_circuit:
            circuit = self._construct_ws_circuit()
        
        # calculate MG Matrix 
        MG = self.gem.get_gem_matrix(circuit, self._min_eigen_solver.quantum_instance)
        
        # set MG to quantum instance
        self._min_eigen_solver.quantum_instance.MG = MG
    
    def solve(self, problem: QuadraticProgram):
        
        #print("solve_with_gem")
    
        self._verify_compatibility(problem)

        # convert problem to QUBO
        problem_ = self._convert(problem, self._converters)

        # construct operator and offset
        operator, offset = problem_.to_ising()
        
        # set gem_matrix to quantum_instance
        self._set_gem_matrix(operator)
        
        return self._solve_internal(operator, offset, problem_, problem)
  