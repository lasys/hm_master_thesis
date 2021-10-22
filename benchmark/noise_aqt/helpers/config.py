from enum import Enum
import numpy as np
import qiskit
from qiskit.providers.aer.noise import NoiseModel
from shared.graph_helper import load_nx_graph_from, generate_butterfly_graph

class Backend(Enum):
    Simulator = 1
    Simulator_Noise_Model = 2
    IBMQ_Toronto = 3

# Optimizer 
MAX_ITER = 0
EVAL_NUM = 50
MAX_P = 4

# Backend 
SHOTS= 200
BACKEND = Backend.Simulator_Noise_Model

# Beta and Gammas

all_initial_points = {
    "graph_05_06_02": [
        np.array([3.74802914, 5.97548754]),
        np.array([ 5.53587131,  4.42587486, -2.92704858, -4.06325998]),
        np.array([-4.83079014, -4.65248254, -4.19891311, -0.27566298,  3.68090215, 1.39655185]),
        np.array([-5.29739445,  0.05296981,  0.45663691,  6.30629268,  2.40931855, -6.01205104, -0.14923894, -3.80925229]),
    ],
    
    "graph_05_06_02_w": [
        np.array([-3.33412359,  5.45892711]),
        np.array([ 0.47267939,  3.07283929, -3.80560601,  3.86218767]),
        np.array([-6.74717392, -0.9163878 , -4.77892508,  1.93667299, -3.17040873, -3.50351501]),
        np.array([ 6.75570487,  1.09724577, -3.19311042,  1.02429928, -4.3558831 , -4.42997004, -4.2206179 , -4.84945794]),
    ],
    
    "graph_3_reg_04_06_01": [
        np.array([-0.54695145,  3.42723337]),
        np.array([ 3.71737342,  2.39708849, -1.32968676,  3.69795058]),
        np.array([ 4.15965249, -2.47254763,  1.68762414, -5.90270808,  3.23038001, -3.89943837]),
        np.array([ 4.70845172, -1.28075305,  1.36848696, -1.68131045,  2.98882816, -3.67267433, -4.64769237, -1.04975779]),
    ],
    
    "graph_3_reg_04_06_w_01": [
        np.array([-2.6072489 ,  4.09683645]),
        np.array([-3.40186785,  2.28061272,  5.75356476, -1.10844472]),
        np.array([ 3.65033485,  5.37269681,  0.39263765, -1.78727933,  0.64890894, 0.45923593]),
        np.array([-5.70082804, -2.00474185,  2.53676052,  1.01981993,  2.15783159, 3.68665365,  4.92940721, -0.44183973]),
    ],
    
    "graph_3_reg_06_09_01": [
        np.array([ 2.58551375, -0.32644157]),
        np.array([-0.47864488,  6.75635959,  5.36804382,  0.27130277]),
        np.array([ 5.64498347,  5.67846441,  3.44034387,  0.46969992, -1.40506581, -3.38533329]),
        np.array([ 5.8625679 , -4.18519437, -3.97184907,  5.94240359,  2.14109226, 4.89165864,  2.45352326, -6.32049763]),
    ],
    
    "graph_3_reg_06_09_w_01": [
        np.array([-6.40641905,  3.51058934]),
        np.array([ 6.39102365, -2.0102223 ,  0.17922266, -0.15558668]),
        np.array([0.08221298,  1.25588222, -3.22868516,  1.57785919,  3.36940372, 6.08556824]),
        np.array([-6.11763842,  1.51108935,  5.97264758,  0.36039349,  2.53809168, 5.14240758, -6.27497456,  2.67537927]),
    ],

    "graph_3_reg_08_12_01": [
        np.array([ 3.70844412, -1.19896617]),
        np.array([ 0.55991658, -0.45390528,  4.12618609,  4.98247732]),
        np.array([3.6037268 , 2.02071539, 0.95239728, 5.02494468, 1.06812494, 4.9127652 ]),
        np.array([-0.44577081,  0.55797946, -1.12040671,  4.44204276, -2.88375301, 5.60043149, -4.0385551 , -4.52369346]),
    ],
    
    "graph_3_reg_08_12_w_01": [
        np.array([-6.17846382,  5.91107867]),
        np.array([4.83421424, 4.77252699, 1.5631106 , 1.17796741]),
        np.array([6.2646854 , -4.55709362, -6.1557145 ,  4.35617418,  0.4798657 , 3.14678775]),
        np.array([ 0.11771653,  4.37735006,  3.51603281,  4.75554454,  2.49946099, -6.07918402, -0.21035086, -6.15134875]),
    ],
    
    "graph_3_reg_10_15_01": [
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    ],
    
    "graph_3_reg_10_15_w_01": [
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    ],
    
}

initial_points = []

def load_configs(name):
    global initial_points
    initial_points = all_initial_points[name].copy()
    return BACKEND, EVAL_NUM, MAX_ITER, MAX_P, SHOTS, initial_points

def display_configs():
    print(f"Backend = {BACKEND.name}")
    print(f"EVAL_NUM = {EVAL_NUM}")
    print(f"MAX_ITER = {MAX_ITER}")
    print(f"MAX_P = {MAX_P}")
    print(f"SHOTS = {SHOTS}")
    print(f"Initial_Points = {initial_points}")
    

    