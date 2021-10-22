import networkx as nx
import numpy as np
import glob
from sys import platform as _platform


def generate_butterfly_graph(with_weights=True):
    # Generate a graph of 5 nodes
    # filename = graph_05_06_02_w.txt
    n = 5
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(0, n, 1))
    if with_weights:
        elist = [(0, 3, 9), (0, 4, 6), (1, 2, 9), (1, 4, 10), (2, 4, 7), (3, 4, 7)]
        graph.name = "graph_05_06_02_w"
    else:
        elist = [(0, 3, 1), (0, 4, 1), (1, 2, 1), (1, 4, 1), (2, 4, 1), (3, 4, 1)]
        graph.name = "graph_05_06_02"
    graph.add_weighted_edges_from(elist)
    
    return graph


def draw_graph(graph):
    labels = nx.get_edge_attributes(graph, 'weight')
    layout = nx.spring_layout(graph)
    nx.draw(graph, pos=layout, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=labels)


GRAPH_FILES_PATH = "/Users/lachermeier/PycharmProjects/master_thesis_qaoa/data/graphs/*_nodes"
GRAPH_FILES_PATH_LINUX = "/home/hm-tlacherm/qlm_notebooks/notebooks_1.2.1/notebooks/masterthesis/data/graphs/*_nodes"
#GRAPH_FILES_PATH_LINUX = "/home/lachermeier/master_thesis_qaoa/data/graphs/*_nodes"

def load_graph_dirs():
    if _platform == "linux":
        return sorted(glob.glob(GRAPH_FILES_PATH_LINUX))

    return sorted(glob.glob(GRAPH_FILES_PATH))


def load_graph_files(path):
    return sorted(glob.glob(path + "/*.txt"))


def load_graphs(specific_nodes=None):
    if specific_nodes is None:
        specific_nodes = []
    dirs = load_graph_dirs()
    graphs = []
    for dir in dirs:
        graphs_in_dir = []
        graph_files = load_graph_files(dir)
        for graph_file in graph_files:
            graph = load_nx_graph_from(graph_file)
            if len(graph.nodes) not in specific_nodes and len(specific_nodes) != 0:
                continue
            graphs_in_dir.append(graph)
        graphs.append((dir.replace('graphs/', ''), graphs_in_dir.copy()))
    return graphs


def load_nx_graph_from(graph_file):
    file = open(graph_file, 'r')
    lines = file.readlines()
    number_of_nodes = int(lines[0])
    lines.pop(0)
    edges = []
    for line in lines:
        elements = line.split(',')
        edges.append((int(elements[0]) - 1, int(elements[1]) - 1, float(elements[2])))

    name = graph_file.replace('.txt', '')
    name = name.split('/')[-1]
    nodes = list(range(0, number_of_nodes))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    graph.name = name

    return graph
