import numpy as np
import networkx as nx
from itertools import product
import numba as nb
import copy
from scipy import optimize

#
# http://127.0.0.1:8888/notebooks/qlm_notebooks/notebooks_1.2.1/notebooks/master_thesis_qaoa/ibm/TQA.ipynb 
#
# Fast Hadamard transformation

@nb.njit # Just in time compilation for massive speed-up
def fwht(a):
    h = 1
    tmp = a.copy()
    while 2 * h <= len(a):
        for i in range(0, len(a) - 1, h * 2):
            for j in range(i, i + h):
                x = tmp[j]
                y = tmp[j + h]
                tmp[j] = x + y
                tmp[j + h] = x - y
        h *= 2
    return tmp

def ifwht(a):
    return fwht(a) / len(a)

# Classical Hamiltonian, the ground state is the MaxCut solution
# we use the QAOA to find an approximation to the ground state variationally
def H_C():
    
    if is_weighted:
        return H_C_weighted()

    tmp = np.zeros(2**N)
    for i, j in edges:
        tmp += H[:, i] * H[:, j]
    return tmp

# Quantum Hamiltonian, the ground state is an equal superposition of all solutions, i.e. all possible graph cuts
def H_B():
    tmp = np.zeros(2**N)
    for i in nodes:
        tmp += H[:, i]
    return tmp

def init_weights(N, seed):

    np.random.seed(seed)
    a = np.random.rand(N, N)
    return np.tril(a) + np.tril(a, -1).T

def H_C_weighted():

    tmp = np.zeros(2**N)
    for i, j in edges:
        tmp += weights[i, j] * H[:, i] * H[:, j]
    return tmp

def qaoa_state(x):

    # Create the QAOA ansatz state, note that we use a fast Hadamard 
    # transformation (n log(n)) to keep the operators diagonal and use vector-vector 
    # multiplication rather than matrix-vector mutliplication where the 
    # matrix would be the matrix exponential of H_B which would be very slow

    beta, gamma = np.split(x, 2)
    state = copy.copy(initial_state)

    for g, b in zip(gamma, beta):

        state = np.exp(1j * g * H_C()) * state
        state = fwht(state)              # Fast Hadamard transformation
        state = np.exp(-1j * b * H_B()) * state
        state = ifwht(state)             # inverse Fast Hadamard transformation

    return state

def energy_expectation_value(state):
    return np.real(np.vdot(state, H_C() * state))

def calculate_tqa(graph, p, print_info=True):
    global edges, nodes, initial_state, H, N, weights, is_weighted
    
    is_weighted = False 
    edges = nx.edge_betweenness(graph).keys()
    nodes = graph.nodes
    N = len(nodes)
    initial_state = np.ones(2 ** N) / np.sqrt(2) ** N
    # Create full Hilbertspace to use in creation of the Hamiltonians
    H = np.array(list(product([1, -1], repeat=N)))
    
    for u,v,w in graph.edges(data=True):
        if w['weight'] != 1:
            # from paper: 
            weights = init_weights(N, 42)
            #is_weighted = True 
            # my idea:
            # weights = nx.adjacency_matrix(graph).todense()
            #print(weights)
            is_weighted=True
            break
    
    # Loop over different evolution times
    time = np.linspace(0.1, N+p, 200)
    
    energy = []
    for t_max in time: 
        dt = t_max / p
        t = dt * (np.arange(1, p + 1) - 0.5)
        gamma = (t / t_max) * dt
        beta = (1 - (t / t_max)) * dt
        x = np.concatenate((beta, gamma))
        energy.append(energy_expectation_value(qaoa_state(x)))
        
    # Find optimal time 
    idx = np.argmin(energy)
    t_max = time[idx]
    # Fix initial beta and gamma values
    dt = t_max / p
    t = dt * (np.arange(1, p + 1) - 0.5)
    gamma = (t / t_max) * dt
    beta = (1 - (t / t_max)) * dt
    
    #tqa_initial_points = np.concatenate([beta,gamma])
    tqa_initial_points = np.concatenate([gamma,beta])
    print(f"TQA: Beta: {beta}, Gamma: {gamma} (p={p})")
    
    return tqa_initial_points 
    


    
    