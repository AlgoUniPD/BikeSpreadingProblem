import os
import itertools
import time as t
import pandas as pd
import numpy as np
import geojson
import json


def read_input(file_name):
    '''
    Read input parameters
    @param file_name: path to the file Input.txt
    @return a dictionary with the input parameters
    '''

    param = pd.read_csv(file_name)
    param_dict = {'graph_filename': param['graph_name'][0],
                  'algorithm': param['algorithm'][0],
                  'model': param['model'][0],
                  'seed_size': param['seed_size'][0],
                  'tau': param['tau'][0],
                  'n_bikes': param['n_bikes'][0],
                  't_model_threshold': param['t_model_threshold'][0],
                  }
    return param_dict


def create_node_list(edges):
    '''
    Starting from list of edges, create the list of nodes
    @param edges: list of edges of the graph
    @return  the list of nodes of the graph
    '''
    nodes = pd.DataFrame(edges, columns=['start', 'end', 'prob'])
    nodes = pd.concat([nodes.start, nodes.end]).drop_duplicates().tolist()
    nodes.sort()
    return nodes


def get_adjacency_matrix(edges, nodes, tau):
    '''
    Create the adjacency matrix of the graph, when considering tau steps
    @param edges: list of edges of the graph
    @param nodes: list of nodes of the graph
    @param tau: number of steps a bike can perform
    @return the adjacency matrix
    '''
    n = len(nodes)
    adj = np.zeros((n, n))
    for e in edges:
        # mappo le n celle nei nodi, ottengo una matrice più piccola
        adj[nodes.index(int(e[0])), nodes.index(int(e[1]))] = e[2]
    adj = np.linalg.matrix_power(adj, tau)
    return adj


def solve_problem(nodes, param_dict, budget, adj):
    '''
    @param nodes: list of nodes of the graph
    @param param_dict: the input parameters on which to perform the filter
    @param budget: the starting budget of bikes in each node of the seed set
    @param adj: adjacency matrix of the graph
    @return a dataframe with the result obtained by the algorithm
    '''

    alg = param_dict['algorithm']
    model = param_dict['model']
    seed_size = param_dict['seed_size']
    c = param_dict['t_model_threshold']
    tau = param_dict['tau']
    start = t.time()
    if (alg == 'greedy'):
        seed, l_best, sigma = greedy(model, seed_size, budget, adj, c)
    if (alg == 'brute'):
        seed, l_best, sigma = brute_force(model, seed_size, budget, adj, c)
    end = t.time()
    my_seed = []
    for i in range(len(seed)):
        my_seed.append(nodes[seed[i]])
    my_seed.sort()

    my_list = [alg, model + '-model', my_seed, round(sigma, 3), c, (end - start) * 1000]
    result_df = pd.DataFrame(data=[my_list],
                             columns=['Algorithm', 'Model', 'Seed', 'Sigma', 'Threshold(t-model)', 'time [ms]'])

    # uncomment to write geojson of the distribution
    get_geojson(my_seed, l_best, nodes, param_dict['graph_filename'])

    return result_df


def compute_sigma(model, v, c):
    '''
    @param model:the model for which to compute the objective function sigma
    @param v: the vector representing the current distribution
    @param c: the t-model_threshold
    @return the value of the objective function for the current distribution
    '''
    if model == 't':
        sigma = 0
        for i in range(len(v)):
            if v[i] >= c:
                sigma += 1
    else:
        sigma = (np.sqrt(v)).sum()

    return sigma


def brute_force(model, seed_size, budget, adj, c):
    '''
    @param model: model used(t or s)
    @param seed_size: size of the seed set
    @param budget: budget of bikes for each node of the seed set
    @param adj: adjacency matrix
    @param c: threshold for t-model
    @return the best seed set found, the value of the objective function and the obtained distribution with such seed
    '''
    n = len(adj[0])
    seed = np.zeros((seed_size,), dtype=int)
    sigma = 0
    l_best = np.zeros((n,))
    # generate all the possible seed sets of size seed_size
    all_vertexes = np.arange(n)
    all_combinations = list(itertools.combinations(all_vertexes, seed_size))
    count = 0
    # for each possible seed set
    for subset in all_combinations:
        count += 1
        l0 = np.zeros((n,), dtype=int)
        for i in range(seed_size):
            l0[subset[i]] = budget
        l_tau = np.matmul(l0, adj)

        current_sigma = compute_sigma(model, l_tau, c)

        if current_sigma > sigma:
            l_best = l_tau
            seed = subset
            sigma = current_sigma

    return seed, l_best, sigma


def greedy(model, seed_size, budget, adj, c):
    '''
    @param model: model used(t or s)
    @param seed_size: size of the seed set
    @param budget: budget of bikes for each node of the seed set
    @param adj: adjacency matrix
    @param c: threshold for t-model
    @return the best seed set found, the value of the objective function and the obtained distribution with such seed
    '''
    n = len(adj[0])
    seed = np.zeros((seed_size,), dtype=int)
    for i in range(seed_size):
        seed[i] = -1
    current_sigma = 0
    l_best = np.zeros((n,))
    for i in range(seed_size):
        best_node, l_best, current_sigma = compute_influence(model, seed, current_sigma, adj, budget, c)
        if best_node == -1:
            print("err: No node improve the current situation.\n")
        seed[i] = best_node

    return seed, l_best, current_sigma


def compute_influence(model, seed, current_sigma, adj, budget, c):
    '''
    @param model: model used(t or s)
    @param seed: current seed set
    @paramm current_sigma: value of the objective function with current seed set
    @param adj: adjacency matrix
    @param budget: budget of bikes for each node of the seed set
    @param c: threshold for t-model
    @return the best node to insert in the seed set, the distribution with such seed and the new value of the objective function
    '''
    max_increment = 0
    best_node = -1
    n = len(adj[0])
    l0 = np.zeros((n,), dtype=float)
    l_best = np.zeros((n,))

    for i in range(len(seed)):
        if not seed[i] == -1:
            l0[seed[i]] = budget

    for i in range(n):
        if not l0[i] == budget:
            l0[i] = budget
            l_tau = np.matmul(l0, adj)
            sigma = compute_sigma(model, l_tau, c)

            if sigma - current_sigma >= max_increment:
                max_increment = sigma - current_sigma
                best_node = i
                l_best = l_tau
            l0[i] = 0
    # update current sigma and seed set
    current_sigma += max_increment

    return best_node, l_best, current_sigma


def get_geojson(seed, v, nodes, graph_name):
    '''
    Writes a geojson file, with the distribution of bikes in the city of Padova, obtained by the algorithm
    @param seed: seed set
    @param v: vector of the distribution of bikes in each node
    @param nodes: list of nodes
    @param graph_name: name of the considered graph
    '''
    grids_path = '../resources/grids'
    output_path = '../output'

    if "500" in graph_name:
        file_name = 'my_grid_500.geojson'
    elif "100" in graph_name:
        file_name = 'my_grid_100.geojson'

    #read the grids
    with open(os.path.join(grids_path, file_name), 'r') as f:
        my_data = json.load(f)

    # set property seed = 1 for nodes in the seed set
    for i in range(len(seed)):
        for feature in my_data['features']:
            if (int(feature['properties']['id']) == seed[i]):
                feature['properties']['seed'] = 1

    # set distribution
    for i in range(len(v)):
        if v[i] > 0.0:
            for feature in my_data['features']:
                if (int(feature['properties']['id']) == nodes[i]):
                    feature['properties']['bikes'] = v[i]

    # write geojson
    with open(os.path.join(output_path, 'final_distribution.geojson'), 'w') as g:
            geojson.dump(my_data, g)
