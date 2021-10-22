from qiskit.providers import Backend, BaseBackend
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.layout import Layout
from qiskit.assembler.run_config import RunConfig
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result
from qiskit.qobj import Qobj
from qiskit import compiler
from qiskit.converters import circuit_to_dag, dag_to_circuit

from .gem_postprocess import format_counts

from typing import Optional, List, Union, Dict, Callable, Tuple
import itertools, copy 
import copy
import logging
from time import time
import numpy as np
import re

class GEMInstance():
    
    def __init__(self):
        self.MG_cache = {}
        self.quantum_instance = None

    def hash_circuit(self, qasm):
        """Input: QASM-String (circuit.qasm()). Returns HashString"""
        gates_str = ""
        for line in qasm.split('\n'):
            if line.startswith('OPEN') or \
                line.startswith('measure') or \
                line.startswith('include') or \
                line.startswith('qreg') or \
                line.startswith('creg'):
                continue
            gates_str += line

        #hashstr = re.sub(r'\((-?([0-9]+.[0-9]*))*\)', '', gates_str)
        hashstr = re.sub(r"\((.*?)\)", '', gates_str)
        #print(gates_str)
        #hashstr = hashstr.replace(' ','')
        #print(hashstr)

        return str( hash(hashstr) )

    def split(self, dag_qc, layers):
        for layer in layers:
            for node in layer:
                if node.type == 'op':
                    dag_qc.remove_op_node(node)

        return dag_to_circuit(dag_qc)


    def split_circuit(self, qc):
        layers = list(circuit_to_dag(qc).multigraph_layers())

        # calculate separation point 
        n_remove = 0
        if len(layers) % 2 == 0: # even
            n_remove = len(layers) // 2
        else: # odd 
            n_remove = (len(layers) - 1) // 2

        # split circuits
        # https://quantumcomputing.stackexchange.com/a/13824/16704 : 
        # the extra minus 1 since the last layer consists of output nodes (qubits and clbits).
        first_half_qc = self.split(circuit_to_dag(qc), layers[- n_remove -1 :])
        second_half_qc = self.split(circuit_to_dag(qc), layers[:- n_remove -1])

        return first_half_qc, second_half_qc
    
    def inverse_and_compose(self, circuit):
        return circuit.copy().compose(circuit.inverse())
    
    
    def prepare_circuits(self, n_qubits):
        start_states = list(map(list, itertools.product([0, 1], repeat=n_qubits)))
        circuits = []
        for state in start_states:
            circuit = QuantumCircuit(n_qubits, n_qubits)
            for i,v in enumerate(state):
                if v == 1:
                    circuit.x(i)

            circuits.append(circuit)

        return circuits.copy()
    
    def compose_circuits(self, prepared, inversed):
        qcs = []
        for pc in prepared:
            qc = pc.copy().compose(inversed)
            qc.measure_all()
            qcs.append(qc.copy())
        
        return qcs
    

    def calculate_freq(self, result):
        n = len(result.header.clbit_labels)//2
        shots = result.shots
        combinations = list(map(list, itertools.product([0, 1], repeat=n)))
        freqs = {}
        for comb in combinations: # init dict
            freqs[re.sub('[^01]', '', str(comb))] = 0.0

        counts = format_counts(result.data.counts, result.header)
        for count in counts.items():
            bitstr, values = count
            if (bitstr == '0'):
                bitstr = '0'*n
            bitstr = bitstr.replace(' ','')
            bitstr = bitstr[:n][::-1]
            freqs[bitstr] = values / shots

        return list(freqs.values())
    
    def create_m(self, results):
        
        m = np.expand_dims( np.array(self.calculate_freq(results[0])), axis=1)
        for result in results[1:]:
            v = np.expand_dims( np.array(self.calculate_freq(result)), axis=1)
            m = np.append(m, v, axis=1)
        return m
    
    def create_m_matrix(self, circuit, n_qubits, had_transpiled) -> np.array:
         # inverse and compose each circuit
        inversed_circuit = self.inverse_and_compose(circuit)
    
        # prepare calibration circuits
        prepared_circuits  = self.prepare_circuits(n_qubits)

        # compose prepared with inversed ones
        calibration_circuits = self.compose_circuits(prepared_circuits, inversed_circuit)
        
        # execute calibration circuits
        result = self.quantum_instance.execute_calibration_circuits(calibration_circuits, had_transpiled)
        m = self.create_m(result.results)
        
        return m
    

    def _construct_mg(self, circuit, quantum_instance):
        
        if quantum_instance is None:
            raise Exception("NO quantum_instance")
            
        self.quantum_instance = quantum_instance  
        
        # remove measure gates 
        circuit.remove_final_measurements()
        
        n_qubits = circuit.num_qubits

        # split circuit in half
        first_half_qc, second_half_qc = self.split_circuit(circuit)

        # calculate M-Matrices
        had_transpiled = False
        M1 = self.create_m_matrix(first_half_qc, n_qubits, had_transpiled)
        M2 = self.create_m_matrix(second_half_qc, n_qubits, had_transpiled)

        # calculate MG-Matrix
        MG = (M1 + M2) / 2

        return MG.copy()
        
    
    def get_gem_matrix(self, circuit, quantum_instance):
        
        # calculate hash and return MG for hash, if exists
        circuit_hash = self.hash_circuit(circuit.qasm())
        
        if circuit_hash not in self.MG_cache:
            #print(f"Matrix for {circuit_hash} not in cache")
            #print(circuit.qasm())
            MG = self._construct_mg(circuit, quantum_instance)
            self.MG_cache[circuit_hash] = MG.copy()
        #else:
        #    print(f"Matrix for {circuit_hash} in cache")
            
        return self.MG_cache[circuit_hash]