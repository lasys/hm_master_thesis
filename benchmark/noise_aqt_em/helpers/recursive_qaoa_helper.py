from qiskit_optimization.algorithms import RecursiveMinimumEigenOptimizer, MinimumEigenOptimizer
from .qaoa_helper import *

def run_recursive(max_cut, qaoa, print_output=True):
    algorithm = MinimumEigenOptimizer(qaoa)
    optimizer = RecursiveMinimumEigenOptimizer(algorithm)
    result = optimizer.solve(max_cut.to_qubo())
    mean, distribution = max_cut.analyse(result, print_output=print_output)
    if print_output:
        max_cut.plot_histogram(distribution, mean)
    

def _run_recursive_evaluation(max_cut, qaoa):
    algorithm = MinimumEigenOptimizer(qaoa)
    optimizer = RecursiveMinimumEigenOptimizer(algorithm)
    result = optimizer.solve(max_cut.to_qubo())
    mean, r, ar = max_cut.analyse_evaluation(result, print_output=False)
    return mean, r, ar


def start_recursive_evaluation(max_cut, eval_num, reps, maxiter=50, init_points=None):
    
    means = []
    ratios = []
    approx_ratios = []
    print(f"p={reps}: ",end='')
    for i in range(0, eval_num):
        qaoa = create_qaoa(optimizer=COBYLA(maxiter=maxiter), reps=reps, initial_point=init_points)
        try:
            mean,r,ar = _run_recursive_evaluation(max_cut, qaoa)
            means.append(mean)
            ratios.append(r)
            approx_ratios.append(ar)
        except Exception as e:
            print(e)
            try:
                mean,r,ar = _run_recursive_evaluation(max_cut, qaoa)
                means.append(mean)
                ratios.append(r)
                approx_ratios.append(ar)
            except Exception as e:
                print(e)
                print(f"Cannot run evaluation {i} with p={reps}")
        
        print(f".",end='')
    print()
        
    return means, ratios, approx_ratios
