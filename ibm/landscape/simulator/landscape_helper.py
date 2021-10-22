import numpy as np
from ibm_landscape_processes import *

step_size = 0.1
gamma_range = np.arange(-np.pi, np.pi, step_size)
beta_range = np.arange(-np.pi, np.pi, step_size)
l_betas, l_gammas = np.meshgrid(beta_range, gamma_range)
landscape = None 


def calculate_landscape_data(filename, max_cut):
    landscape = run_all(l_betas, l_gammas, max_cut)
    # Save result matrix 
    with open(filename, 'wb') as f:
        np.save(f, landscape)

def load_landscape_data(filename, max_cut):
    global landscape
    if os.path.isfile(filename) == False:
        print("Landscape does not exists, start calculating..")
        calculate_landscape_data(filename, max_cut)
        
    f = open(filename, 'rb')
    landscape = np.load(f)
    f.close()
    
    return landscape

def describe_landscape(landscape):
    # Mean of landscape
    mean = np.mean(landscape)
    # Minimium 
    min_exp = np.min(landscape)
    # beta + gamma values 
    # beta + gamma value of Minimium
    i,j = np.unravel_index(np.argmin(landscape), landscape.shape)
    min_gamma = l_gammas[i,j]
    min_beta = l_betas[i,j]
    print(f"Landscape mean: {mean:.2f}")
    print(f"Min Exp.Value: {min_exp:.2f}")
    print(f"Min. Beta: {min_beta:.2f}, Min. Gamma: {min_gamma:.2f}")
    
    return min_beta, min_gamma, min_exp
    