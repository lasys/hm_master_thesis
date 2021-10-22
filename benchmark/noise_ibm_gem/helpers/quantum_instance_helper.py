import qiskit
from qiskit import Aer
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.models import BackendConfiguration
from .config import *
import pickle 
from shared.gem.gem_quantum_instance import GEMQuantumInstance

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
        return GEMQuantumInstance(
                                        backend=Aer.get_backend(DEFAULT_QASM_SIMULATOR),
                                        shots=SHOTS,
                                        )
    
    if BACKEND.value == Backend.Simulator_Noise_Model.value:
        quantum_instance = _create_simulator_with_noise_quantum_instance()
    elif BACKEND.value == Backend.IBMQ_Toronto.value:
        raise Exception("Quantum Instance with Toronto not implemented!")
    else: 
        raise Exception(f"Quantum Instance not instanciated! {BACKEND.value}")
    
    return quantum_instance


def _init_noise_model_parameters():
    global noise_model, coupling_map, basis_gates
    if noise_model is None or coupling_map is None or basis_gates is None:
        noise_model_filename = f"../helpers/ibmq_toronto_noise_model_dict.pkl"
        noise_model_dict = load_from_pickle(noise_model_filename)
        noise_model = NoiseModel.from_dict(noise_model_dict)
        basis_gates = noise_model.basis_gates
        
        noise_backend_configuration_filename = f"../helpers/ibmq_toronto_noise_backend_configuation_dict.pkl"
        noise_backend_configuration_dict = load_from_pickle(noise_backend_configuration_filename)
        noise_backend_configuration = BackendConfiguration.from_dict(noise_backend_configuration_dict)
        coupling_map = noise_backend_configuration.coupling_map
        

def _create_simulator_with_noise_quantum_instance():
    
    _init_noise_model_parameters()
    
    quantum_instance = GEMQuantumInstance(
                                        backend=Aer.get_backend(DEFAULT_QASM_SIMULATOR),
                                        shots=SHOTS,
                                        noise_model=noise_model,
                                        coupling_map=coupling_map,
                                        basis_gates=basis_gates,
                                        )
    return quantum_instance


def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
