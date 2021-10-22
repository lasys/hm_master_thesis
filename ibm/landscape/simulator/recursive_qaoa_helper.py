from qiskit_optimization.algorithms import RecursiveMinimumEigenOptimizer,MinimumEigenOptimizer, IntermediateResult
from qaoa_helper import *


def run_recursive(max_cut, qaoa=None, print_output=True):
    if qaoa is None:
        qaoa = create_qaoa()
    algorithm = MinimumEigenOptimizer(qaoa)
    optimizer = RecursiveMinimumEigenOptimizer(algorithm,min_num_vars_optimizer=algorithm, history=IntermediateResult.ALL_ITERATIONS)
    result = optimizer.solve(max_cut.to_qubo())
    mean, distribution = max_cut.analyse(result, print_output=print_output)
    if print_output:
        max_cut.plot_histogram(distribution, mean)
    

def run_recursive_evaluation(qaoa, max_cut):
    algorithm = MinimumEigenOptimizer(qaoa)
    optimizer = RecursiveMinimumEigenOptimizer(algorithm, min_num_vars_optimizer=algorithm, history=IntermediateResult.ALL_ITERATIONS)
    
    result = optimizer.solve(max_cut.to_qubo())
    print(qaoa.ansatz) 
    mean, r, ar = max_cut.analyse_evaluation(result, print_output=False)
    return mean, r, ar


def start_recursive_evaluation(max_cut, eval_num, reps, init_points=None):
    
    means = []
    ratios = []
    approx_ratios = []
    print(f"p={reps}: ",end='')
    for i in range(0, eval_num):
        qaoa = create_qaoa(optimizer=COBYLA(maxiter=50), reps=reps, initial_point=init_points)
        try:
            mean,r,ar = run_recursive_evaluation(qaoa, max_cut)
            means.append(mean)
            ratios.append(r)
            approx_ratios.append(ar)
        except:
            try:
                mean,r,ar = run_recursive_evaluation(qaoa, max_cut)
                means.append(mean)
                ratios.append(r)
                approx_ratios.append(ar)
            except:
                print(f"Cannot run evaluation {i} with p={reps}")
        
        print(f".",end='')
    print()
        
    return means, ratios, approx_ratios
