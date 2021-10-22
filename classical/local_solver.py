import random
from itertools import combinations, groupby
import localsolver
import networkx as nx

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G


def calculate_local_solver_solution(nodes, edges):
    with localsolver.LocalSolver() as ls:
        ls.param.set_verbosity(0)
        number_of_nodes = len(nodes)
        number_of_edges = len(edges)

        # Origin of each edge
        origin = [None]*number_of_edges
        # Destination of each edge
        dest = [None]*number_of_edges
        # Weight of each edge
        weights = [None]*number_of_edges
        e = 0
        for (u, v, w) in edges:
            origin[e] = u + 1
            dest[e] = v + 1
            weights[e] = w
            e += 1

        # Declares the optimization model
        model = ls.model

        # Decision variables x[i]
        # Is true if vertex x[i] is on the right side of the cut and false if it is on the left side of the cut
        x = [model.bool() for i in range(number_of_nodes)]

        # incut[e] is true if its endpoints are in different class of the partition
        incut = [None]*number_of_edges
        for e in range(number_of_edges):
            incut[e] = model.neq(x[origin[e] - 1], x[dest[e] - 1])

        # Size of the cut
        cut_weight = model.sum(weights[e]*incut[e] for e in range(number_of_edges))
        model.maximize(cut_weight)
        model.close()

        # Param
        #ls.param.time_limit = 10
        ls.solve()

        highest_cut = cut_weight.value
        solution_bitstring = [k.value for k in x]

        return highest_cut, solution_bitstring