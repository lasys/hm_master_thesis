from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, CobylaOptimizer, MinimumEigenOptimizer, GroverOptimizer,  SlsqpOptimizer
from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer
from qaoa_helper import *
from plot_helper import *
import matplotlib.pyplot as plt

# todo: change backend 
def run_ws_qaoa(max_cut, qaoa, epsilon=0.25,print_output=True):
    
    ws_qaoa = WarmStartQAOAOptimizer(pre_solver=GoemansWilliamsonOptimizer(5), relax_for_pre_solver=False, qaoa=qaoa, epsilon=epsilon)
    result = ws_qaoa.solve(max_cut.to_qubo())
    optimal_parameters = qaoa.optimal_params
    mean, distribution = max_cut.analyse(result, print_output=print_output)
    if print_output:
        print(f"Optimal Parameters: {optimal_parameters % 3.14}")
        print(f"Run WarmStartQAOAOptimizer with epsilon: {epsilon}")
        max_cut.plot_histogram(distribution, mean)
        
    return result, mean, optimal_parameters

def run_epsilon_evaluation_ws_qaoa(max_cut, qaoa, epsilon=0.25, print_output=True):
    result, mean, optimal_parameters = run_ws_qaoa(max_cut, qaoa, epsilon, print_output=print_output)
    found_max_cut = result.fval == max_cut.opt_max_cut
    
    return mean, found_max_cut


def optimize_epsilon(max_cut, reps=1, print_output=False):
    means = []
    eps = []
    for e in range(100,-1, -1):
        epsi = e*0.005
        # only add if opt_maxt_cut is found  
        qaoa = create_qaoa(reps=reps)
        mean, found = run_epsilon_evaluation_ws_qaoa(max_cut, qaoa, epsilon=epsi, print_output=print_output)
        if found: 
            means.append(mean)
            eps.append(epsi)
    
    plt.plot(eps, means)
    plt.xlabel('ɛ')
    plt.ylabel('expectation value')
    plt.show()
    opt_eps = eps[ np.argmin(means) ]
    print(f"ɛ={opt_eps}")
    
    return opt_eps


def run_evaluation_ws_qaoa(max_cut, qaoa, epsilon=0.25, print_output=False):
    result, mean, optimal_parameters = run_ws_qaoa(max_cut, qaoa, epsilon, print_output=print_output)
    mean, r, ar = max_cut.analyse_evaluation(result, print_output=False)
    return mean, r, ar


def start_ws_qaoa_evaluation(max_cut, eval_num, reps, epsilon):
    
    means = []
    ratios = []
    approx_ratios = []
    print(f"p={reps}: ",end='')
    for i in range(0, eval_num):
        qaoa = create_qaoa(optimizer=COBYLA(maxiter=50), reps=reps)
        try:
            mean,r,ar = run_evaluation_ws_qaoa(max_cut, qaoa, epsilon=epsilon)
            means.append(mean)
            ratios.append(r)
            approx_ratios.append(ar)
        except:
            try:
                mean,r,ar = run_evaluation_ws_qaoa(max_cut, qaoa, epsilon=epsilon)
                means.append(mean)
                ratios.append(r)
                approx_ratios.append(ar)
            except:
                print(f"Cannot run evaluation {i} with p={reps}")
        
        print(f".",end='')
    print()
    
    return means, ratios, approx_ratios

        