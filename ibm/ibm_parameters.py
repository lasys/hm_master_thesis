import math
import qiskit
from qiskit.algorithms.optimizers import SPSA, COBYLA, QNSPSA
from qiskit.providers.aer.noise import NoiseModel
from qiskit_optimization.algorithms import OptimizationResult

from shared.graph_helper import load_nx_graph_from, generate_butterfly_graph

MAX_ITER = 5
optimizers = {
    #"ADAM": ADAM(maxiter=MAX_ITER), # benötigt extrem lange
    #"AQGD": AQGD(maxiter=MAX_ITER), # benötigt extrem lange
    "SPSA": SPSA(maxiter=MAX_ITER),
    "QN-SPSA": "QN-SPSA",
    "COBYLA": COBYLA(maxiter=MAX_ITER),
    #"NELDER_MEAD": NELDER_MEAD(maxiter=MAX_ITER),
}

# Optuna parameters
REPS_MAX = 5
REPS_MIN = 1
GAMMA_MIN = -math.pi
GAMMA_MAX = math.pi
BETA_MIN = -math.pi
BETA_MAX = math.pi
N_TRIALS = 5

# Noise
BACKEND_NAME = 'ibmq_toronto'
provider = qiskit.IBMQ.get_provider(hub='ibm-q-unibw', group='hochschule-muc', project='masterarbeit')
noise_backend = provider.get_backend(BACKEND_NAME)
noise_model = NoiseModel.from_backend(noise_backend)
coupling_map = noise_backend.configuration().coupling_map
basis_gates = noise_model.basis_gates

SHOTS= 8000

DEFAULT_QASM_SIMULATOR='qasm_simulator'
#DEFAULT_QASM_SIMULATOR='ibmq_qasm_simulator'

def save_best_trial(study, trial):
    if study.best_trial.number == trial.number:
        mean, result, optimal_parameters, optimizer_name = trial.user_attrs["best"]
        result_copy = OptimizationResult(result.x.copy(), result.fval.copy(), result.variables.copy(), status=result.status, samples=result.samples.copy())
        study.set_user_attr(key="best", value=[mean,result_copy, optimal_parameters.copy(), optimizer_name])


def print_parameters():
    print(f"Optimizers: {list(optimizers.keys())} with MaxIter of {MAX_ITER}")
    print(f"Number of shots: {SHOTS}")
    print(f"Repetitions: [ {REPS_MIN}; {REPS_MAX} ]")
    print(f"Gamma value interval: [ {GAMMA_MIN}; {GAMMA_MAX} ]")
    print(f"Beta value interval: [ {BETA_MAX};{BETA_MIN} ]")
    print(f"Number of Optuna Trials: {N_TRIALS}")
    print(f"Noise Backend Name: {BACKEND_NAME}")

def load_graph():
    graph = generate_butterfly_graph(with_weights=False)
    # paper graph:
    #graph = load_nx_graph_from("../../data/graphs/16_nodes/graph_16_33_01_w.txt")
    return graph
