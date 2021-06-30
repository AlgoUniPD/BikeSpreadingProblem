# ## IMPORT

import pandas as pd
import os

from utils import (
    read_input,
    get_adjacency_matrix,
    solve_problem,
    create_node_list
)

resources_path = '../resources'
output_path = '../output'
# read all parameters from Input.txt
param_dict = read_input(os.path.join(resources_path, 'input.csv'))

# budget of bikes in each node of seed set
budget = int(param_dict['n_bikes'] / param_dict['seed_size'])

# ##COMPUTE GRAPH AND ITS ADJAENCY MATRIX

graph_filename = param_dict['graph_filename']
graph_path = os.path.join(resources_path, 'graphs', graph_filename)
edges_df = pd.read_csv(graph_path)

edges = []
for index, row in edges_df.iterrows():
    single_edge_with_weigth = [int(row[0]), int(row[1]), row[2]]
    edges.append(single_edge_with_weigth)

nodes = create_node_list(edges)

adj = get_adjacency_matrix(edges, nodes, param_dict['tau'])

# ##SOLVE THE MODEL
df = solve_problem(nodes, param_dict, budget, adj)

# ##WRITE OUTPUT
df.to_csv(os.path.join(output_path, 'results.csv'), index=False)

