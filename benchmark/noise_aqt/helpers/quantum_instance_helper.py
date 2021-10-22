import qiskit
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.models import BackendConfiguration
from .config import *
import pickle 

from qiskit_aqt_provider import AQTProvider
aqt = AQTProvider('6f47670f3e5c414da0cdcab1c048eb97')
noise_simulator_backend = aqt.backends.aqt_qasm_simulator_noise_1


# create instance depending on config 
def create_quantum_instance():
    from .config import BACKEND

    quantum_instance = None
    
    if BACKEND.value == Backend.Simulator_Noise_Model.value:
        quantum_instance = _create_noise_simulator_quantum_instance()
        
    else: 
        raise Exception(f"Quantum Instance not instanciated! {BACKEND.value}")
    
    return quantum_instance


def _create_noise_simulator_quantum_instance():
    quantum_instance = QuantumInstance(
                           backend=noise_simulator_backend,
                           shots=200,
                       )
    return quantum_instance
