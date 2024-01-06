import numpy as np
import pandas as pd
import time
import random
from itertools import combinations

def create_problem_instance(path):
    metadata = pd.read_csv(path, sep=" ", nrows=1, header=None).iloc[0]
    s = metadata.iloc[0]
    n = metadata.iloc[1]
    m = metadata.iloc[2]

    df = pd.read_csv(path, sep=" ", skiprows=1, names = ["n1", "n2", "e", "w"])

    nodes_from = df.loc[df["e"]==1][["n1","w"]].groupby(['n1']).sum()
    nodes_to = df.loc[df["e"]==1][["n2","w"]].groupby(['n2']).sum()

    nodes = nodes_from.join(nodes_to, lsuffix='_from', rsuffix='_to', how = 'outer')
    nodes['node_impact'] = nodes.w_from.fillna(0) + nodes.w_to.fillna(0)
    nodes = nodes.drop(columns=['w_from', 'w_to'])
    nodes['current_degree'] = 0
    nodes['splex'] = nodes.index
    nodes = nodes.reset_index().rename(columns={"index":"node_number"})
    
    node_impact = nodes["node_impact"].copy().to_numpy()
    node_degree = nodes["current_degree"].copy().to_numpy()
    plex_assignment = nodes["splex"].copy().to_numpy()

    edges = df.copy()
    edges['w'] = edges['w'] * (1-(edges['e']*2))
    edges['e'] = 0
    
    edge_weights = edges["w"].copy().to_numpy()
    edge_assignment = edges["e"].copy().to_numpy()

    return nodes, node_impact, node_degree, plex_assignment, edges, edge_weights, edge_assignment, s, n, m
    
def is_splex(node_degree, plex_assignment, plex_number, s) -> bool | np.ndarray:
    
    min_degree = sum(plex_assignment==plex_number) - s
    problem_nodes = np.where((node_degree < min_degree) & (plex_assignment == plex_number))[0]+1
    
    if len(problem_nodes) == 0:
        return True
    else:
        return problem_nodes

def get_edge_index(a, b, n) -> int:
    if a != b:
        if(a>b):
            tmp = a
            a = b
            b = tmp
        return int((a-1)*n-((a-1)*a)/2+b-a-1)
    else:
        raise ValueError("both nodes have the same index")

        
def add_good_edges(node_impact, node_degree, plex_assignment, edge_weights, edge_assignment, plex_number):
    # get all nodes from that plex
    nodes_in_plex = np.where(plex_assignment == plex_number)[0]+1
    # get all edges (i.e. node combinations) from that plex
    edge_combs = list(combinations(nodes_in_plex, 2))
    # get their edge_index
    edge_index_tmp = np.fromiter((get_edge_index(b[0],b[1],len(node_impact)) for b in edge_combs), "int64")
    # only get the ones that are not yet used and have a negative/neutral weight
    edge_index = edge_index_tmp[(edge_assignment[edge_index_tmp]==0) & (edge_weights[edge_index_tmp]<=0)]
    
    if len(edge_index)>0: #only need to do the next steps if there are actually edges to add
        # get corresponding node number, so we can adjust the node info later on
        edge_combs = [
            edge_combs[i] for i in range(len(edge_combs)) 
            if edge_assignment[edge_index_tmp][i] == 0 and edge_weights[edge_index_tmp][i] <= 0
        ]
        # split the edge_combinations into from and to nodes for easier handling
        from_nodes = np.array([combo[0] for combo in edge_combs])
        to_nodes = np.array([combo[1] for combo in edge_combs])

        ### add them to plex ###
        # change edge_assignment 
        edge_assignment[edge_index]=1
        # change node_degree of both ends of the edge
        np.add.at(node_degree, (from_nodes-1), 1)
        np.add.at(node_degree, (to_nodes-1), 1)
        # change node_impact of both ends of the edge
        np.add.at(node_impact, (from_nodes-1), edge_weights[edge_index])
        np.add.at(node_impact, (to_nodes-1), edge_weights[edge_index])
        

def repair_splex(node_impact, node_degree, plex_assignment, edge_weights, edge_assignment, plex_number, s, problem_nodes):
    nodes_in_plex = np.where(plex_assignment == plex_number)[0]+1
    # shuffe problem nodes, so they are not always repaired in the same order
    np.random.shuffle(problem_nodes)
    for node in problem_nodes:
        # see how many edges we need to add
        edges_needed = sum(plex_assignment==plex_number) - s - node_degree[node-1] # min_degree - curent degree
        if edges_needed <1:
            continue # the node degree necessary was already reached when another nodes degree was repaired
        
        ### get all potential edges ###
        # remove the current node from the nodes_in_plex to get potential neighbors
        potential_neighbor_nodes = np.delete(nodes_in_plex, np.where(nodes_in_plex == node))
        # get index of all possible edges for that node within that plex
        edge_index_tmp = np.fromiter((get_edge_index(node,b,len(node_impact)) for b in potential_neighbor_nodes), 
                                 potential_neighbor_nodes.dtype)
        # only get the ones that are not yet used
        edge_index = edge_index_tmp[edge_assignment[edge_index_tmp]==0]
        # and also their corresponding node number, so we can adjust the node info later on
        potential_neighbor_nodes = potential_neighbor_nodes[edge_assignment[edge_index_tmp]==0]
        
        ### get the cheapes edges ###
        # i.e. from all potential edges (of which we have edge_index), get the ones with smallest weight
        # apparently the fastest way is argpartition https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        order_of_cheapest_edges = np.argpartition(edge_weights[edge_index], edges_needed)
        # these are the indices of the cheapest edges from our current node to another one within the plex (and the nodes)
        cheapest_edges_index = edge_index[order_of_cheapest_edges[:edges_needed]]
        cheapest_nodes = potential_neighbor_nodes[order_of_cheapest_edges[:edges_needed]]

        ### add them to plex ###
        # change edge_assignment 
        edge_assignment[cheapest_edges_index]=1
        # change node_degree of neighbors
        node_degree[(cheapest_nodes-1)] += 1
        #change node degree of current node
        node_degree[node-1] += edges_needed
        # change node_impact of neighbors
        node_impact[(cheapest_nodes-1)] += edge_weights[cheapest_edges_index]
        # change node_impact of current node  
        node_impact[node-1] += sum(edge_weights[cheapest_edges_index])
        
    #return node_impact, node_degree, edge_assignment # don't need to return as python modifies the arrays

def repair_solution(node_impact, node_degree, plex_assignment, edge_weights, edge_assignment, s):
    # get all plexes
    all_plexes = np.unique(plex_assignment)
    
    for plex in all_plexes:
        # add all edges we want to have anyway
        add_good_edges(node_impact, node_degree, plex_assignment, edge_weights, edge_assignment, plex_number = plex)
        # check if it is a valid splex
        valid_plex = is_splex(node_degree, plex_assignment, plex_number = plex, s=s)
        if not isinstance(valid_plex, (bool)):
            repair_splex(node_impact, node_degree, plex_assignment, 
                         edge_weights, edge_assignment, plex_number=plex, s=s, problem_nodes=valid_plex)