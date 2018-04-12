import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import linear_model

def open_file_data_graph(file_path):
    with open(file_path, encoding="utf8") as data:
        data_list = []
        for line in data.readlines():
            line = line.strip("\n")
            line = line.split(" ")
            line = [int(item) for item in line]
            line.append({"weight": 1})
            data_list.append(line)
        data.close()
    graph = nx.Graph()
    graph.add_edges_from(data_list)
    node_number = graph.number_of_nodes()
    edge_number = graph.number_of_edges()
    return graph, node_number, edge_number


def demo_graph_generator():
    # undirected graph uses nx.Graph()
    # directed graph uses nx.Digraph()
    graph = nx.Graph()
    graph.add_edges_from([[0, 1, {"weight": 1}],
                          [0, 2, {"weight": 1}],
                          [0, 4, {"weight": 1}],
                          [0, 5, {"weight": 1}],
                          [0, 7, {"weight": 1}],
                          [1, 4, {"weight": 1}],
                          [1, 5, {"weight": 1}],
                          [1, 6, {"weight": 1}],
                          [2, 3, {"weight": 1}],
                          [2, 6, {"weight": 1}],
                          [3, 4, {"weight": 1}],
                          [3, 6, {"weight": 1}],
                          [4, 5, {"weight": 1}],
                          [4, 6, {"weight": 1}],
                          [5, 6, {"weight": 1}],
                          [5, 7, {"weight": 1}]])

    node_number = len(graph.nodes())
    edge_number = len(graph.edges())
    return graph, node_number, edge_number


def game_array(r, s, t, p):
    return np.array([[r, s], [t, p]])


def calculate_gain(graph, g_array, strategy):
    gain_dict = {}
    for node in graph:
        gain_dict[node] = 0.0
    for node in graph:
        adj_nodes = nx.all_neighbors(graph,node)
        for adj_node in adj_nodes:
            gain_dict[node] += multi_dot(strategy[node],g_array,np.transpose(strategy[adj_node]))
    return gain_dict


def init_strategy(graph):
    strategy_dict = dict()
    for node in graph:
        if np.random.random() < 0.5:
            strategy_dict[node] = np.array([1,0])
        else:
            strategy_dict[node] = np.array([0,1])
    return strategy_dict


def multi_dot(*c):
    result = c[0]
    for i in range(1, len(c)):
        result = np.dot(result, c[i])
    return result


def random_choose(graph, node):
    # return a integer of the number of node
    neighbors = list(nx.all_neighbors(graph,node))
    return random.sample(neighbors, 1)[0]


def whether_change_strategy(graph, node,gain_dict, k=0.1):
    compare_node = random_choose(graph,node)
    prob = 1/(1+(np.exp((gain_dict[node] - gain_dict[compare_node])/k)))
    if random.random() < prob:
        return True
    else:
        return False


def change_strategy(strategy, node):
    return np.array([1-strategy[node][0],1-strategy[node][1]])


def evolutionary_data(cas):
    #demo_graph, node_number, edge_number = demo_graph_generator()
    demo_graph, node_number, edge_number = open_file_data_graph("data/smallworld.txt")
    g_array = game_array(1, 0.3, 1.7, 0)
    strategy_dict = init_strategy(demo_graph)
    evolve_strategy = list()
    evolve_gain = list()
    for i in range(cas):
        gain_dict = calculate_gain(demo_graph, g_array, strategy_dict)
        # record the strategy_dict and gain data
        evolve_strategy.append(strategy_dict.copy())
        evolve_gain.append(gain_dict.copy())
        for node in demo_graph.nodes():
            if whether_change_strategy(demo_graph,node,gain_dict,k=0.01):
                strategy_dict[node] = change_strategy(strategy_dict,node)

    generate_A_G(demo_graph,g_array,evolve_strategy,evolve_gain,3)
    return evolve_strategy, evolve_gain


def generate_A_G(graph, g_array, evolve_strategy,evolve_gain, node):
    other_nodes = list(graph.nodes())
    other_nodes.remove(node)
    other_nodes = list(sorted(other_nodes))
    mapping_dict = dict()
    print(other_nodes)
    print(list(nx.all_neighbors(graph,node)))
    for i in range(len(other_nodes)):
        mapping_dict[i] = other_nodes[i]
    row = len(evolve_strategy)
    col = len(other_nodes)
    node_matrix = [[0.0 for i in range(col)] for j in range(row)]
    gain_array = [0.0 for i in range(row)]
    for i in range(row):
        for j in range(col):
            node_matrix[i][j] = multi_dot(evolve_strategy[i][node], g_array, np.transpose(evolve_strategy[i][mapping_dict[j]]))
        gain_array[i] = evolve_gain[i][node]
    clf = linear_model.Lasso(alpha=0.05)
    clf.fit(node_matrix,gain_array)
    print(clf)
    print(clf.coef_)
    return node_matrix, gain_array


def show_strategy(graph,evolve_strategy):
    pass


if __name__ == "__main__":
    evolve_strategy, evolve_gain = evolutionary_data(70)