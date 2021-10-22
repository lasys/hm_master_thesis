import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))

import qiskit
#provider = qiskit.IBMQ.load_account()
from qiskit.algorithms.optimizers import SPSA, COBYLA, QNSPSA
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
#from ibm.ibm_parameters import *
from multiprocessing import Process, Value, Array
import multiprocessing as mp
from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer
from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, CobylaOptimizer, MinimumEigenOptimizer, SlsqpOptimizer

#BACKEND_NAME = 'ibmq_toronto'
#provider = qiskit.IBMQ.get_provider(hub='ibm-q-unibw', group='hochschule-muc', project='masterarbeit')
#noise_backend = provider.get_backend(BACKEND_NAME)
#noise_model = NoiseModel.from_backend(noise_backend)
#coupling_map = noise_backend.configuration().coupling_map
#basis_gates = noise_model.basis_gates

#
# simulator
#


def run_qaoa(beta, gamma, max_cut):
    quantum_instance = QuantumInstance(
                    backend=Aer.get_backend("qasm_simulator"),
                    shots=8000)
    
    qaoa = QAOA(optimizer=COBYLA(maxiter=0),
                quantum_instance=quantum_instance,
                reps=1,
                initial_point=[beta, gamma])
    
    max_cut_qubo = max_cut.to_qubo()
    algorithm = MinimumEigenOptimizer(qaoa)
    result = algorithm.solve(max_cut_qubo)
    optimal_parameters = qaoa.optimal_params
    mean, distribution = max_cut.analyse(result)

    return mean

def run_qaoa_process(arr, index, beta, gamma, max_cut):
    mean = run_qaoa(beta, gamma, max_cut)
    arr[index] = mean


def run_all(b_beta,a_gamma, max_cut):
    mp.freeze_support()
    ctx = mp.get_context('spawn')

    f1_temp = np.zeros(b_beta.shape)
    for i in range(0, len(f1_temp)):
        processes = []
        arr = ctx.Array('f', range(len(f1_temp)))
        for j in range(0, len(f1_temp)):
            p = ctx.Process(target=run_qaoa_process, args=(arr, j, b_beta[i][j], a_gamma[i][j], max_cut))
            p.start()
            processes.append(p)
    
        for pp in processes:
            pp.join()
        
        for j in range(0,len(arr)):
            f1_temp[i,j] = arr[j]
           
        print(f"Row {i}")
            
    return f1_temp.copy()


#
# Simulator with noise 
#

def run_qaoa_noise(beta, gamma, max_cut):
    quantum_instance_noise = QuantumInstance(
                backend=Aer.get_backend(DEFAULT_QASM_SIMULATOR),
                shots=SHOTS,
                noise_model=noise_model,
                coupling_map=coupling_map,
                basis_gates=basis_gates
    )
    
    qaoa = QAOA(optimizer=COBYLA(maxiter=0),
                quantum_instance=quantum_instance_noise,
                reps=1,
                initial_point=[beta, gamma])
    
    max_cut_qubo = max_cut.to_qubo()
    algorithm = MinimumEigenOptimizer(qaoa)
    result = algorithm.solve(max_cut_qubo)
    optimal_parameters = qaoa.optimal_params
    mean, distribution = max_cut.analyse(result)

    return mean

def run_qaoa_process_noise(arr, index, beta,gamma, max_cut):
    mean = run_qaoa_noise(beta, gamma, max_cut)
    arr[index] = mean


def run_all_noise(b_beta, a_gamma, max_cut):
    mp.freeze_support()
    ctx = mp.get_context('spawn')

    f1_temp = np.zeros(b_beta.shape)
    for i in range(0, len(f1_temp)):
        processes = []
        arr = ctx.Array('f', range(len(f1_temp)))
        for j in range(0, len(f1_temp)):
            p = ctx.Process(target=run_qaoa_process_noise, args=(arr, j, b_beta[i][j], a_gamma[i][j], max_cut))
            p.start()
            processes.append(p)
    
        for pp in processes:
            pp.join()
        
        for j in range(0,len(arr)):
            f1_temp[i,j] = arr[j]
           
        print(f"Row {i}")
            
    return f1_temp.copy()

#
# Warmstart
#

def run_qaoa_warmstart(beta, gamma, max_cut):
    quantum_instance = QuantumInstance(
                    backend=Aer.get_backend(DEFAULT_QASM_SIMULATOR),
                    shots=SHOTS)
    
    qaoa = QAOA(optimizer=COBYLA(maxiter=0),
                quantum_instance=quantum_instance,
                reps=1,
                initial_point=[beta, gamma])
    
    ws_qaoa = WarmStartQAOAOptimizer(pre_solver=GoemansWilliamsonOptimizer(), relax_for_pre_solver=False,
                                 qaoa=qaoa, epsilon=0.01)
    
    max_cut_qubo = max_cut.to_qubo()
    result = ws_qaoa.solve(max_cut_qubo)
    optimal_parameters = qaoa.optimal_params
    mean, distribution = max_cut.analyse(result)

    return mean

def run_qaoa_process_warmstart(arr, index, beta, gamma, max_cut):
    mean = run_qaoa_warmstart(beta, gamma, max_cut)
    arr[index] = mean

def run_all_warmstart(b_beta, a_gamma, max_cut):
    mp.freeze_support()
    ctx = mp.get_context('spawn')

    f1_temp = np.zeros(b_beta.shape)
    for i in range(0, len(f1_temp)):
        processes = []
        arr = ctx.Array('f', range(len(f1_temp)))
        for j in range(0, len(f1_temp)):
            p = ctx.Process(target=run_qaoa_process_warmstart, args=(arr, j, b_beta[i][j], a_gamma[i][j], max_cut))
            p.start()
            processes.append(p)
    
        for pp in processes:
            pp.join()
        
        for j in range(0,len(arr)):
            f1_temp[i,j] = arr[j]
           
        print(f"Row {i}")
            
    return f1_temp.copy()
