from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from shared.QiskitMaxcut import *
from .quantum_instance_helper import create_quantum_instance
from shared.cdr.cdr_qaoa import CDR_QAOA
from shared.ws_qaoa import WS_QAOA
import numpy as np
#
# QAOA 
#
def create_qaoa(optimizer=COBYLA(maxiter=0), quantum_instance=None, reps=1, initial_point=None): 
    if quantum_instance is None: 
        quantum_instance = create_quantum_instance()
    
    qaoa = CDR_QAOA(optimizer = optimizer,
            quantum_instance=quantum_instance,
            reps=reps,
            initial_point=initial_point,
           )
    
    return qaoa

def create_ws_qaoa(optimizer=COBYLA(maxiter=0), quantum_instance=None, reps=1, initial_point=None): 
    if quantum_instance is None: 
        quantum_instance = create_quantum_instance()
    
    init_p = [np.pi/2] + [0.] * reps
    
    qaoa = WS_QAOA(optimizer = optimizer,
            quantum_instance=quantum_instance,
            reps=reps,
            initial_point=init_p,
           )
    
    return qaoa


def run_qaoa(qaoa, qubo):
    algorithm = MinimumEigenOptimizer(qaoa)
    result = algorithm.solve(qubo)
    optimal_parameters = qaoa.optimal_params
    return result, optimal_parameters


def _run_qaoa_evaluation(max_cut, qaoa):
    algorithm = MinimumEigenOptimizer(qaoa)
    result = algorithm.solve(max_cut.to_qubo())
    mean, r, ar = max_cut.analyse_evaluation(result, print_output=False)
    return mean, r, ar
    
def start_qaoa_evaluation(max_cut, eval_num, reps, maxiter=50, init_points=None):
    means = []
    ratios = []
    approx_ratios = []
    print(f"p={reps}: ",end='')
    for i in range(0, eval_num):
        qaoa = create_qaoa(optimizer=COBYLA(maxiter=maxiter),reps=reps, initial_point=init_points)
        try:
            mean,r,ar = _run_qaoa_evaluation(max_cut,qaoa)
            means.append(mean)
            ratios.append(r)
            approx_ratios.append(ar)
        except Exception as e:
            print(e)
            try:
                mean,r,ar = _run_qaoa_evaluation(max_cut, qaoa)
                means.append(mean)
                ratios.append(r)
                approx_ratios.append(ar)
            except Exception as e:
                print(e)
                print(f"Cannot run evaluation {i} with p={reps}")
        
        print(f".",end='')
    print()
    
    return means, ratios, approx_ratios

