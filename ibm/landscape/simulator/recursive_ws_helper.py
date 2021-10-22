from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, CobylaOptimizer, MinimumEigenOptimizer, GroverOptimizer,  SlsqpOptimizer
from qiskit_optimization.algorithms import RecursiveMinimumEigenOptimizer,MinimumEigenOptimizer, IntermediateResult
from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer

from qaoa_helper import *
import matplotlib.pyplot as plt

def run_recursive_ws_qaoa(max_cut, qaoa, epsilon=0.25, print_output=False):
    ws_qaoa = WarmStartQAOAOptimizer(pre_solver=GoemansWilliamsonOptimizer(5), relax_for_pre_solver=False, qaoa=qaoa, epsilon=epsilon)
    optimizer = RecursiveMinimumEigenOptimizer(ws_qaoa, history=IntermediateResult.ALL_ITERATIONS)
    result = optimizer.solve(max_cut.to_qubo())
    optimal_parameters = qaoa.optimal_params
    mean, distribution = max_cut.analyse(result, print_output=print_output)
    if print_output:
        print(f"Optimal Parameters: {optimal_parameters % 3.14}")
        print(f"Run Recursive WarmStartQAOAOptimizer with epsilon: {epsilon}")
        max_cut.plot_histogram(distribution, mean)
        
    return result, mean, optimal_parameters

def run_evaluation_recursive_ws_qaoa(max_cut, qaoa, epsilon=0.25, print_output=False):
    ws_qaoa = WarmStartQAOAOptimizer(pre_solver=GoemansWilliamsonOptimizer(5), relax_for_pre_solver=False, qaoa=qaoa, epsilon=epsilon)
    optimizer = RecursiveMinimumEigenOptimizer(ws_qaoa, history=IntermediateResult.ALL_ITERATIONS)
    result = optimizer.solve(max_cut.to_qubo())
    mean, r, ar = max_cut.analyse_evaluation(result, print_output=print_output)
    return mean, r, ar


def start_recursive_ws_qaoa_evaluation(max_cut, eval_num, reps, epsilon):
    
    means = []
    ratios = []
    approx_ratios = []
    print(f"p={reps}: ",end='')
    for i in range(0, eval_num):
        qaoa = create_qaoa(reps=reps, optimizer=COBYLA(maxiter=50))
        try:
            mean,r,ar = run_evaluation_recursive_ws_qaoa(max_cut, qaoa=qaoa, epsilon=epsilon)
            means.append(mean)
            ratios.append(r)
            approx_ratios.append(ar)
        except:
            try:
                mean,r,ar = run_evaluation_recursive_ws_qaoa(max_cut, qaoa=qaoa, epsilon=epsilon)
                means.append(mean)
                ratios.append(r)
                approx_ratios.append(ar)
            except:
                print(f"Cannot run evaluation {i} with p={reps}")
                
        
        print(f".",end='')
    print()
    
    return means, ratios, approx_ratios

        