import qiskit
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.models import BackendConfiguration
from .config import *
import pickle 

from qiskit_ionq import IonQProvider
provider = IonQProvider("MPRTyJJN82dEDlmeACadX3VdOLl4DFmv")
# Get IonQ's simulator backend:
simulator_backend = provider.get_backend("ionq_simulator")

#
# Quantum Instance 
#
DEFAULT_QASM_SIMULATOR = "qasm_simulator"
NOISE_BACKEND = "ibmq_toronto"

# Noise 
noise_model = None 
coupling_map = None
basis_gates = None

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
                           backend=simulator_backend,
                           shots=SHOTS,
                       )
    return quantum_instance

