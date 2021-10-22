from enum import Enum

import qiskit
from qiskit.providers.aer.noise import NoiseModel
from shared.graph_helper import load_nx_graph_from, generate_butterfly_graph

class Backend(Enum):
    Simulator = 1
    Simulator_Noise_Model = 2
    IBMQ_Toronto = 3

# Optimizer 
MAX_ITER = 5
EVAL_NUM = 10
MAX_P = 4

# Backend 
SHOTS= 200
BACKEND = Backend.Simulator

def load_configs():
    return BACKEND, EVAL_NUM, MAX_ITER, MAX_P, SHOTS

def display_configs():
    print(f"Backend = {BACKEND.name}")
    print(f"EVAL_NUM = {EVAL_NUM}")
    print(f"MAX_ITER = {MAX_ITER}")
    print(f"MAX_P = {MAX_P}")
    print(f"SHOTS = {SHOTS}")

    