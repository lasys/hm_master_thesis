import numpy as np
import qiskit
# provider = qiskit.IBMQ.load_account()
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.algorithms import QAOA
from shared.QiskitMaxcut import *
from ibm.ibm_parameters import *
from landscape_helper import *
from plot_helper import *

# callback helper 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    energy_values.append(mean)
    b, g = np.array_split(parameters,2)
    optimizer_gammas.append(g[0])
    optimizer_betas.append(b[0])
    # get current landscape value 
    _, g_index = find_nearest(gamma_range, g[0])
    _, b_index = find_nearest(beta_range, b[0])
    if landscape is None:
        print("no landscape")
    maxcut_values.append(landscape[g_index, b_index])
    
    
def calculate_fidelity(pre_qaoa, max_cut_qubo):
    MinimumEigenOptimizer(pre_qaoa).solve(max_cut_qubo)
    fidelity = QNSPSA.get_fidelity(pre_qaoa.ansatz)
    return fidelity

def create_quantum_instance():
    quantum_instance = QuantumInstance(
                                        backend=Aer.get_backend(DEFAULT_QASM_SIMULATOR),
                                        shots=SHOTS
                                       )
    return quantum_instance

def create_quantum_instace_with_error_mitigation():
    return QuantumInstance(
                    backend=Aer.get_backend(DEFAULT_QASM_SIMULATOR),
                    shots=SHOTS,
                    measurement_error_mitigation_cls=CompleteMeasFitter,
                    measurement_error_mitigation_shots=SHOTS)

def create_ibmq_toronto_quantum_instance():
    return QuantumInstance(
                    backend=provider.get_backend('ibmq_toronto'),
                    shots=SHOTS,
                    measurement_error_mitigation_cls=CompleteMeasFitter,
                    measurement_error_mitigation_shots=SHOTS)


def create_qaoa(optimizer=COBYLA(maxiter=0), quantum_instance=None, reps=1, initial_point=None, with_callback=False): 
    if quantum_instance is None: 
        quantum_instance = create_quantum_instance()
    qaoa = QAOA(optimizer = optimizer,
            quantum_instance=quantum_instance,
            reps=reps,
            initial_point=initial_point,
            callback=store_intermediate_result if with_callback else None
           )
    
    return qaoa

def run_qaoa(qaoa, qubo, with_callback=False):
    algorithm = MinimumEigenOptimizer(qaoa)
    result = algorithm.solve(qubo)
    optimal_parameters = qaoa.optimal_params
    return result, optimal_parameters

def run_qaoa_with_callback(qaoa, qubo):
    global counts, energy_values ,maxcut_values, optimizer_gammas, optimizer_betas
    counts = []
    energy_values = []
    maxcut_values = []
    optimizer_gammas = []
    optimizer_betas = []
        
    algorithm = MinimumEigenOptimizer(qaoa)
    result = algorithm.solve(qubo)
    optimal_parameters = qaoa.optimal_params
    return result, optimal_parameters, ( counts.copy(), energy_values.copy(), maxcut_values.copy(), optimizer_gammas.copy(), optimizer_betas.copy() )

def run_qaoa_evaluation(max_cut, qaoa):
    algorithm = MinimumEigenOptimizer(qaoa)
    result = algorithm.solve(max_cut.to_qubo())
    mean, r, ar = max_cut.analyse_evaluation(result, print_output=False)
    return mean, r, ar
    
    
    # TODO: IBM Parameters vll hier einfügen?? 
    # TODO: maxiter von Optimizer auch von außen herein geben! 
    
    
def start_qaoa_evaluation(max_cut, eval_num, reps, maxiter=50, init_points=None):
    
    means = []
    ratios = []
    approx_ratios = []
    print(f"p={reps}: ",end='')
    for i in range(0, eval_num):
        qaoa = create_qaoa(optimizer=COBYLA(maxiter=maxiter),reps=reps, initial_point=init_points)
        try:
            mean,r,ar = run_qaoa_evaluation(max_cut,qaoa)
            means.append(mean)
            ratios.append(r)
            approx_ratios.append(ar)
        except:
            try:
                mean,r,ar = run_qaoa_evaluation(max_cut, qaoa)
                means.append(mean)
                ratios.append(r)
                approx_ratios.append(ar)
            except:
                print(f"Cannot run evaluation {i} with p={reps}")
        
        print(f".",end='')
    print()
    
    return means, ratios, approx_ratios

