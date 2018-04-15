import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import linear_model
import cvxpy


def open_file_data_graph(file_path):
    # create graph topology from data directory

    with open(file_path, encoding="utf8") as data:
        data_list = []
        for line in data.readlines():
            line = line.strip("\n")
            line = line.split(" ")
            line = [int(item) for item in line]
            line.append({"weight": 1})
            data_list.append(line)
        data.close()
    # create graph use networkx library
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
    # the payoff matrix of the game model
    return np.array([[r, s], [t, p]])


def sg_array(r=0.7):
    return np.array([[1, 1 - r], [1 + r, 0]])


def pdg_array(b=1.2):
    return np.array([[1, 0], [b, 0]])


def calculate_gain(graph, g_array, strategy):
    # In one round, there are strategies and gains of each node
    # The result of gain_dict is a dict of gain of each node in the certain round
    # gain_dict = {node1:gain1,node2:gain2,...}

    # initial
    gain_dict = {}
    for node in graph:
        gain_dict[node] = 0.0
    # accumulate the gain of one node towards other linked nodes
    for node in graph:
        adj_nodes = nx.all_neighbors(graph, node)
        for adj_node in adj_nodes:
            gain_dict[node] += multi_dot(strategy[node], g_array, np.transpose(strategy[adj_node]))
    return gain_dict


def init_strategy(graph):
    # Init the strategy of each node in round 1 randomly
    # cooperate [1,0] or defect[0,1]

    strategy_dict = dict()
    for node in graph:
        if np.random.random() < 0.5:
            strategy_dict[node] = np.array([1, 0])
        else:
            strategy_dict[node] = np.array([0, 1])
    return strategy_dict


def multi_dot(*c):
    # matrix multiplication with two matrices or more

    result = c[0]
    for i in range(1, len(c)):
        result = np.dot(result, c[i])
    return result


def random_choose(graph, node):
    # return a random index of neighbors of the given node

    neighbors = list(nx.all_neighbors(graph, node))
    return random.sample(neighbors, 1)[0]


def whether_change_strategy(graph, node, gain_dict, strategy_dict, k):
    # while one round is over
    # the participators can compare their gain to one random neighbor and choose whether to follow his strategy

    compare_node = random_choose(graph, node)
    prob = 1 / (1 + (np.exp((gain_dict[node] - gain_dict[compare_node]) / k)))
    if random.random() < prob:
        return strategy_dict[compare_node].copy()
    else:
        return strategy_dict[node].copy()


def get_evolutionary_data(graph, g_array, cas, k=0.1):
    # get the strategy and gain data of the game, given g_array and the cascade number
    # evolve_strategy and evolve_gain are two list of the strategy dict and gain dict in each round

    strategy_dict = init_strategy(graph)
    evolve_strategy = list()
    evolve_gain = list()
    # repeat cas times to get the n rounds data, n=cas
    for i in range(cas):
        gain_dict = calculate_gain(graph, g_array, strategy_dict)
        # record the strategy_dict and gain data
        evolve_strategy.append(strategy_dict.copy())
        evolve_gain.append(gain_dict.copy())
        for node in graph.nodes():
            # in each round, each node in the graph changes its strategy according to the probability w
            strategy_dict[node] = whether_change_strategy(graph, node, gain_dict, strategy_dict, k)

    return evolve_strategy, evolve_gain


def generate_adjacent_vector_of_one_node(graph, g_array, evolve_strategy, evolve_gain, node):
    # given graph, game matrix, strategy and gain, reconstruct the adjacent vector of a certain node using l1-norm minimization
    # return a vector of adjacent information of the node

    other_nodes = list(graph.nodes())
    # remove the self loop case
    other_nodes.remove(node)
    other_nodes = list(sorted(other_nodes))
    mapping_dict = dict()
    for i in range(len(other_nodes)):
        mapping_dict[i] = other_nodes[i]
    row = len(evolve_strategy)
    col = len(other_nodes)
    node_matrix = [[0.0 for i in range(col)] for j in range(row)]
    gain_array = [0.0 for i in range(row)]
    for i in range(row):
        for j in range(col):
            node_matrix[i][j] = multi_dot(evolve_strategy[i][node], g_array,
                                          np.transpose(evolve_strategy[i][mapping_dict[j]]))
        gain_array[i] = evolve_gain[i][node]

    reconstructed_adj_nodes = list()

    # Basis Pursuit
    # It is also called l1-norm minimization
    # using external package: cvxpy
    x = cvxpy.Variable(col)
    objective = cvxpy.Minimize(cvxpy.sum_entries(cvxpy.abs(x)))
    input_matrix = np.array(node_matrix)
    input_gain = np.array(gain_array)
    constraints = [input_matrix * x == input_gain]
    problem = cvxpy.Problem(objective, constraints)
    result = problem.solve()
    # low_bound and up_bound are the thresholds of deciding whether the edge exists
    low_bound = 0.8
    up_bound = 1.2
    for i in range(len(x.value)):
        if low_bound < x.value[i] < up_bound:
            reconstructed_adj_nodes.append(mapping_dict[i])

    """
    # Lasso Regression
    clf = linear_model.Lasso(alpha=0.05)
    clf.fit(node_matrix,gain_array)
   
    for i in range(len(clf.coef_)):
        if clf.coef_[i] > 0.5:
            reconstructed_adj_nodes.append(mapping_dict[i])
    """
    return reconstructed_adj_nodes


def reconstruct_graph(graph, g_array, evolve_strategy, evolve_gain):
    # reconstruct the whole graph by summing all the adjacent vectors of all nodes in the graph

    new_graph = nx.Graph()
    for node in graph.nodes():
        adj_nodes = generate_adjacent_vector_of_one_node(graph, g_array, evolve_strategy, evolve_gain, node)
        for adj_node in adj_nodes:
            new_graph.add_edge(node, adj_node)
    return new_graph


def generate_file_name_list():
    # file name of different data for the convenience of automatic iteration

    file_name_list = ["scalefree", "polbooks", "football", "apollonian",
                      "dolphins", "karate", "lattice2d", "miserables",
                      "pseudofractal", "randomgraph", "scalefree", "sierpinski",
                      "smallworld"]
    return file_name_list


def run():
    # output the figure of precision and recall of the reconstruction algorithm

    file_name_list = generate_file_name_list()
    for file_name in file_name_list:
        test(file_name)


def test(file_name):
    # one result of the given several datasets

    file_name_full = "../data/" + file_name + ".txt"
    demo_graph, node_number, edge_number = open_file_data_graph(file_name_full)  # load graph
    print(file_name, "node: ", node_number, "edge: ", edge_number)
    cas_list = [int(i / 20 * node_number) for i in range(1, 21)]  # data amount list for iteration
    REPEAT = 20  # repeat number of each box-plot point, 20 by default
    g_array = sg_array(r=0.7)
    precision_array = np.zeros(shape=(REPEAT, len(cas_list)))  # results
    recall_array = np.zeros(shape=(REPEAT, len(cas_list)))  # results
    for i in range(REPEAT):
        for j in range(len(cas_list)):
            # get the strategy and gain data of one sample
            evolve_strategy, evolve_gain = get_evolutionary_data(graph=demo_graph, g_array=g_array, cas=cas_list[j],
                                                                 k=0.1)
            #  reconstructed
            re_graph = reconstruct_graph(demo_graph, g_array, evolve_strategy, evolve_gain)

            # evaluate the reconstruction effectiveness using precision and recall
            TP = TN = FN = FP = 0
            reconstructed_list = list(re_graph.edges())
            original_list = list(demo_graph.edges())
            all_number = int(node_number * (node_number - 1) / 2)
            for edge in reconstructed_list:
                if (edge[0], edge[1]) in original_list or (edge[1], edge[0]) in original_list:
                    TP += 1
                else:
                    FP += 1
            FN = edge_number - TP
            TN = all_number - TP - FN - FP
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            print("precision: ", precision, "recall: ", recall)
            precision_array[i][j] = precision
            recall_array[i][j] = recall
    # draw box-plot figure
    draw_figure(file_name,precision_array,recall_array)


def draw_figure(file_name,precision_array,recall_array):

    labels = ["0." + str(int((5 * i + 5) / 10)) if i % 2 == 1 else " " for i in range(19)]
    labels.append(str("1.0"))

    plt.figure(figsize=(12, 5))

    # precision box-plot
    plt.subplot(1, 2, 1)
    a = plt.boxplot(precision_array, labels=labels)
    set_box_color(a, '#D7191C')
    plt.ylim((-0.05, 1.05))
    plt.title(file_name)
    plt.xlabel('Data')
    plt.ylabel('Precision')

    # recall box-plot
    plt.subplot(1, 2, 2)
    b = plt.boxplot(recall_array, labels=labels)
    set_box_color(b, '#2C7BB6')
    plt.ylim((-0.05, 1.05))
    plt.title(file_name)
    plt.xlabel('Data')
    plt.ylabel('Recall')

    # save figure
    plt.savefig("../result_figure/" + file_name + ".svg", format="svg")
    plt.savefig("../result_figure/" + file_name + ".png", dpi=600, format="png")
    plt.savefig("../result_figure/" + file_name + ".jpg", dpi=600, format="jpg")


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


if __name__ == "__main__":
    run()
