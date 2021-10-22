import math
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit_optimization.algorithms import OptimizationResult

MAX_ITER = 1
optimizers = {
    #"ADAM": ADAM(maxiter=MAX_ITER), # benötigt extrem lange
    #"AQGD": AQGD(maxiter=MAX_ITER), # benötigt extrem lange
    "SPSA": SPSA(maxiter=MAX_ITER),
    "COBYLA": COBYLA(maxiter=MAX_ITER),
    #"NELDER_MEAD": NELDER_MEAD(maxiter=MAX_ITER),
}

# Optuna parameters
REPS_MAX = 10
REPS_MIN = 1
GAMMA_MIN = -math.pi
GAMMA_MAX = math.pi
BETA_MIN = -math.pi
BETA_MAX = math.pi
N_TRIALS = 2

SHOTS= 200

def save_best_trial(study, trial):
    if study.best_trial.number == trial.number:
        mean, result, optimal_parameters = trial.user_attrs["best"]
        result_copy = OptimizationResult(result.x.copy(), result.fval.copy(), result.variables.copy(), status=result.status, samples=result.samples.copy())
        study.set_user_attr(key="best", value=[mean,result_copy, optimal_parameters.copy()])

