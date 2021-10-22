# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""An application class for the Max-cut."""
import math
from typing import List, Dict, Optional, Union
import networkx as nx
import numpy as np
from docplex.mp.model import Model
from matplotlib import pyplot as plt
from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
# from qiskit_optimization.translators import from_docplex_mp
from shared.docplex_mp import from_docplex_mp
from qiskit_optimization.applications.graph_optimization_application import GraphOptimizationApplication, \
    _HAS_MATPLOTLIB

from qiskit.tools.visualization import plot_histogram


class Maxcut(GraphOptimizationApplication):
    """Optimization application for the "max-cut" [1] problem based on a NetworkX graph.

    References:
        [1]: "Maximum cut",
        https://en.wikipedia.org/wiki/Maximum_cut
    """
    
    def __init__(self, graph, opt_max_cut=None):
        self.opt_max_cut = opt_max_cut
        super().__init__(graph)

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a Max-cut problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the Max-cut problem instance.
        """
        mdl = Model(name="Max-cut")
        x = {
            i: mdl.binary_var(name="x_{0}".format(i)) for i in range(self._graph.number_of_nodes())
        }
        for w, v in self._graph.edges:
            self._graph.edges[w, v].setdefault("weight", 1)
        objective = mdl.sum(
            self._graph.edges[i, j]["weight"] * x[i] * (1 - x[j])
            + self._graph.edges[i, j]["weight"] * x[j] * (1 - x[i])
            for i, j in self._graph.edges
        )
        mdl.maximize(objective)
        op = from_docplex_mp(mdl)
        return op

    def to_qubo(self):
        max_cut_problem = self.to_quadratic_program()
        conv_toQubo = QuadraticProgramToQubo()
        max_cut_qubo = conv_toQubo.convert(max_cut_problem)
        return max_cut_qubo

    def draw(
            self,
            result: Optional[Union[OptimizationResult, np.ndarray]] = None,
            pos: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Draw a graph with the result. When the result is None, draw an original graph without
        colors.

        Args:
            result: The calculated result for the problem
            pos: The positions of nodes
        Raises:
            MissingOptionalLibraryError: if matplotlib is not installed.
        """

        if not _HAS_MATPLOTLIB:
            raise MissingOptionalLibraryError(
                libname="matplotlib",
                name="GraphOptimizationApplication",
                pip_install="pip install 'qiskit-optimization[matplotlib]'",
            )

        labels = nx.get_edge_attributes(self._graph, 'weight')
        layout = nx.spring_layout(self._graph)

        if result is None:
            nx.draw(self._graph, pos=layout, with_labels=True)
        else:
            self._draw_result(result, layout)

        nx.draw_networkx_edge_labels(self._graph, pos=layout, edge_labels=labels)

    def _draw_result(
            self,
            result: Union[OptimizationResult, np.ndarray],
            pos: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Draw the result with colors

        Args:
            result : The calculated result for the problem
            pos: The positions of nodes
        """
        if isinstance(result, np.ndarray):
            x = result
        else:
            x = self._result_to_x(result)

        nx.draw(self._graph,
                with_labels=True, pos=pos,
                node_color=["lightblue" if b == 1 else 'lightgreen' for b in x],
                edge_color=['red' if x[a] != x[b] else 'black' for a, b in self._graph.edges()]
                )

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[List[int]]:
        """Interpret a result as two lists of node indices

        Args:
            result : The calculated result of the problem

        Returns:
            Two lists of node indices correspond to two node sets for the Max-cut
        """
        x = self._result_to_x(result)
        cut = [[], []]  # type: List[List[int]]
        for i, value in enumerate(x):
            if value == 0:
                cut[0].append(i)
            else:
                cut[1].append(i)
        return cut

    def _node_color(self, x: np.ndarray) -> List[str]:
        # Return a list of strings for draw.
        # Color a node with red when the corresponding variable is 1.
        # Otherwise color it with blue.
        return ["r" if value == 0 else "b" for value in x]

    def parse_gset_format(filename: str) -> np.ndarray:
        """Read graph in Gset format from file.

        Args:
            filename: the name of the file.

        Returns:
            An adjacency matrix as a 2D numpy array.
        """
        n = -1
        with open(filename) as infile:
            header = True
            m = -1
            count = 0
            for line in infile:
                # pylint: disable=unnecessary-lambda
                v = map(lambda e: int(e), line.split())
                if header:
                    n, m = v
                    w = np.zeros((n, n))
                    header = False
                else:
                    s__, t__, _ = v
                    s__ -= 1  # adjust 1-index
                    t__ -= 1  # ditto
                    w[s__, t__] = t__
                    count += 1
            assert m == count
        w += w.T
        return w

    def get_gset_result(x: np.ndarray) -> Dict[int, int]:
        """Get graph solution in Gset format from binary string.

        Args:
            x: binary string as numpy array.

        Returns:
            A graph solution in Gset format.
        """
        return {i + 1: 1 - x[i] for i in range(len(x))}

    def analyse(self, result: Union[OptimizationResult, np.ndarray], print_output=False):
        
        # GoemansWilliamsonOptimizer returns positive values -> convert to negative ones
        for i in range(0,len(result.samples)):
            if result.samples[i].fval > 0:
                result.samples[i].fval *= -1
        
        mean = sum(r.fval * r.probability for r in result.samples)
        distribution = self._calc_distribution(result)

        max_v = (max(distribution.keys()), f"{distribution[max(distribution.keys())]:.5}")
        min_v = (min(distribution.keys()), f"{distribution[min(distribution.keys())]:.5}")
        max_p = (max(distribution.keys(), key=(lambda key: distribution[key])),
                 f"{distribution[max(distribution.keys(), key=(lambda key: distribution[key]))]:.5}")
        r = (mean - max_v[0]) / (min_v[0] - max_v[0])
        mse = 0
        for d in list(distribution.keys()):
            mse += (d - mean)**2
        mse /= len(distribution.keys())
        rmse = math.sqrt(mse)
        
        if self.opt_max_cut is None:
            approx_ratio = -1
        else:
            approx_ratio = mean / self.opt_max_cut

        if print_output:
            print(result)
            self._print_formatted_samples(result)
            print(f"Expectation Value: {mean}")
            print(f"Highest Exp.Value: {max_v[0]} with {float(max_v[1]) * 100} %")
            print(f"Lowest Exp.Value: {min_v[0]} with {float(min_v[1]) * 100} %")
            print(f"Highest Probability: {max_p[0]} with {float(max_p[1]) * 100} %")
            print(f"Ratio r: {r}")
            print(f"Approiximation ratio: {approx_ratio}")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")

        return mean, distribution
    
    def analyse_evaluation(self, result: Union[OptimizationResult, np.ndarray], print_output=False):
        
        if self.opt_max_cut is None:
            raise Exception('analyse_evaluation', 'opt_max_cut not set')
            
        # GoemansWilliamsonOptimizer returns positive values -> convert to negative ones
        for i in range(0,len(result.samples)):
            if result.samples[i].fval > 0:
                result.samples[i].fval *= -1
        
        mean = sum(r.fval * r.probability for r in result.samples)
        distribution = self._calc_distribution(result)

        max_v = (max(distribution.keys()), f"{distribution[max(distribution.keys())]:.5}")
        min_v = (min(distribution.keys()), f"{distribution[min(distribution.keys())]:.5}")
        max_p = (max(distribution.keys(), key=(lambda key: distribution[key])),
                 f"{distribution[max(distribution.keys(), key=(lambda key: distribution[key]))]:.5}")
        r = (mean - max_v[0]) / (min_v[0] - max_v[0])
        # if (min_v[0] - max_v[0]) ==0 
        if math.isnan(r):
            r = 0
        mse = 0
        for d in list(distribution.keys()):
            mse += (d - mean)**2
        mse /= len(distribution.keys())
        rmse = math.sqrt(mse)
        
        approx_ratio = mean / self.opt_max_cut

        if print_output:
            print(result)
            self._print_formatted_samples(result)
            print(f"Expectation Value: {mean}")
            print(f"Highest Exp.Value: {max_v[0]} with {float(max_v[1]) * 100} %")
            print(f"Lowest Exp.Value: {min_v[0]} with {float(min_v[1]) * 100} %")
            print(f"Highest Probability: {max_p[0]} with {float(max_p[1]) * 100} %")
            print(f"Ratio r: {np.round(r, 3)}")
            print(f"Approiximation ratio: {np.round(approx_ratio, 3)}")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")

        return mean, r, approx_ratio

    def _calc_distribution(self, result):
        distribution = {}
        for sample in result.samples:
            val = int(sample.fval)
            if sample.probability <= 0.0:
                continue
            if val in distribution.keys():
                distribution[val] += sample.probability
            else:
                distribution[val] = sample.probability
        return distribution

    def _print_formatted_samples(self, result):
        if len(result.samples) > 10:
            print(f"Number of samples ({len(result.samples)}) is too large to display. Skip.")
        else:
            formatted_samples = self.format_qaoa_samples(result.samples)
            for fs in formatted_samples:
                print(fs)

    def format_qaoa_samples(self, samples, max_len: int = 1000):
        qaoa_res = []
        for s in samples:
            qaoa_res.append((''.join([str(int(_)) for _ in s.x]), s.fval, s.probability))

            res = sorted(qaoa_res, key=lambda x: -x[1])[0:max_len]

        return [(_[0] + f': value: {_[1]:.3f}, probability: {1e2 * _[2]:.1f}%') for _ in res]

    def plot_histogram(self, distribution, mean):
        if len(distribution.keys()) > 20:
            from matplotlib.ticker import FixedLocator, FixedFormatter
            fig = plot_histogram(distribution, title=f"Distribution (Ø: {mean:.3f})", bar_labels=False)
            ax = fig.axes[0]
            x_locator = FixedLocator([0, len(distribution.keys())//2, len(distribution.keys())])
            x_formatter = FixedFormatter([
                ""+str(min(distribution.keys())), ""+str(list(distribution.keys())[len(distribution.keys())//2]), ""+str(max(distribution.keys()))])
            ax.xaxis.set_major_locator(x_locator)
            ax.xaxis.set_major_formatter(x_formatter)
            plt.show()
        else:
            _=plot_histogram(distribution, title=f"Distribution (Ø: {mean:.3f})")
            plt.show()

    #
    # DWave
    #
    def calculate_Q_matrix(self):

        max_cut_problem = self.to_quadratic_program()
        mac_cut_qubo = QuadraticProgramToQubo().convert(max_cut_problem)

        # print(mac_cut_qubo)

        quad_coef = mac_cut_qubo.objective.quadratic.coefficients.toarray().copy()
        linear_coef = mac_cut_qubo.objective.linear.coefficients.toarray().copy()
        n = len(linear_coef[0])
        for i in range(n):
            quad_coef[i, i] = linear_coef[0][i]

        # print(quad_coef)
        Q = {}
        for i in range(n):
            for j in range(n):
                if quad_coef[i, j] != 0:
                    Q[i, j] = quad_coef[i, j]

        # print(mac_cut_qubo.objective)

        return Q.copy()



## #%% if histo is ugly:
# from matplotlib.ticker import AutoMinorLocator, FixedLocator, FixedFormatter
# fig = plot_histogram(distribution, bar_labels=len(distribution.keys())<10 )
# ax = fig.axes[0]
# #ax.set_xlim(0, 100)
# t = [-273, -88]
# x_locator = FixedLocator([0, len(distribution.keys())//2, len(distribution.keys())])
# x_formatter = FixedFormatter([
#     ""+str(min(distribution.keys())), "middle", ""+str(max(distribution.keys()))])
# ax.xaxis.set_major_locator(x_locator)
# ax.xaxis.set_major_formatter(x_formatter)
# #ax.xaxis.set_minor_locator(AutoMinorLocator())
# #ax.set_xticks(np.arange(-273, -88, -1))