import qiskit
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.models import BackendConfiguration
from .config import *
import pickle 

#
# Quantum Instance 
#
DEFAULT_QASM_SIMULATOR = "qasm_simulator"


# create instance depending on config 
def create_quantum_instance():
    from .config import BACKEND

    quantum_instance = None
    
    if BACKEND.value == Backend.Simulator.value:
        quantum_instance = _create_simulator_quantum_instance()
    else: 
        raise Exception(f"Quantum Instance not instanciated! {BACKEND.value}")
    
    return quantum_instance


def _create_simulator_quantum_instance():
    quantum_instance = QuantumInstance(
                           backend=Aer.get_backend(DEFAULT_QASM_SIMULATOR),
                           shots=SHOTS,
                       )
    return quantum_instance



def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
