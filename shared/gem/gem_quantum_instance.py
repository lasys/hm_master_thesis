import itertools, copy 
import copy
import logging
from time import time
import numpy as np
import re

from scipy.optimize import minimize
from typing import Optional, List, Union, Dict, Callable, Tuple

from qiskit.providers import Backend, BaseBackend
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.layout import Layout
from qiskit.assembler.run_config import RunConfig
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result
from qiskit.qobj import Qobj
from qiskit import compiler
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.utils import QuantumInstance


from .gem_postprocess import format_counts

class GEMQuantumInstance(QuantumInstance):
        
    def get_freq_vector(self, result):
        counts = result.data.counts.copy()
        counts = format_counts(counts, result.header)
        n = len(result.header.clbit_labels)
        shots = self._run_config.shots
        
        # es kann sein, dass nicht immer alle combis in counts sind, deswegen: 
        combinations = list(map(list, itertools.product([0, 1], repeat=n)))
        freqs = {}
        for comb in combinations: # init dict
            freqs[re.sub('[^01]', '', str(comb))] = 0
        
        v = []
        for key in sorted(counts):
            freqs[key] = counts[key] / shots
            
        for key in sorted(freqs):
            v.append( freqs[key] )
    
        return v.copy()
        
    
    def apply_gem(self, v, MG):
        # scipy minimize 
        constraints = (
            {'type': 'eq', 'fun': lambda x: sum(x) - 1},
        )
        bounds = tuple([(0,1) for _ in range(0,len(v))])

        
        X = np.random.rand(len(v))
        
        # minimize cost function 
        def cost_function(X, V, M):
            cost = 0
            MX = M @ X  
            for i in range(0, len(V)):
                v = V[i]
                cost += ( v - MX[i] ) ** 2
            return cost

        res = minimize(cost_function, X, method='SLSQP', args=(v, MG), 
                       constraints=constraints, bounds=bounds,
                       options={'gtol': 1e-10, 'disp': False})

        return res.x
    
    def convert_to_counts(self, freqs, counts):
        
        freqs = freqs * self._run_config.shots
        #i = 0
        for key in sorted(counts):
            #counts[key] = int( np.round(freqs[int(key, 16)]) )
            counts[key] = freqs[int(key, 16)]
         #   i  += 1
        return counts

    def execute_calibration_circuits(self,
                circuits: Union[QuantumCircuit, List[QuantumCircuit]],
                had_transpiled: bool = False) -> Result:
        
        return super().execute(circuits, had_transpiled)
        
    
    def execute(self,
                circuits: Union[QuantumCircuit, List[QuantumCircuit]],
                had_transpiled: bool = False) -> Result:
        
       # raise Exception("f!")
        
        if self.MG is None:
            raise Exception("GEM-Matrix is None!")
        

        
        result = super().execute(circuits, had_transpiled)
        try: 
            for i in range(0, len(circuits)):
                v = self.get_freq_vector(result.results[i])

               # print(v)
                
                # apply gem and get mitigated v 
                x = self.apply_gem(v, self.MG)

                # convert freqs to counts 
                mitigated_counts = self.convert_to_counts(x, result.results[i].data.counts.copy())

                # replace previous counts with mitigated counts 
                result.results[i].data.counts = mitigated_counts
            
        except Exception as e:
            print(e)
            print(result)
            raise Exception(e)

        return result