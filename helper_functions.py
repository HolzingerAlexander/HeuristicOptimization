import numpy as np
import pandas as pd
import time
import random

def create_problem_instance(path):
    metadata = pd.read_csv(path, sep=" ", nrows=1, header=None).iloc[0]
    s = metadata.iloc[0]

    df = pd.read_csv(path, sep=" ", skiprows=1, names = ["n1", "n2", "e", "w"])

    nodes_from = df.loc[df["e"]==1][["n1","w"]].groupby(['n1']).sum()
    nodes_from
    nodes_to = df.loc[df["e"]==1][["n2","w"]].groupby(['n2']).sum()
    nodes_to

    nodes = nodes_from.join(nodes_to, lsuffix='_from', rsuffix='_to', how = 'outer')
    nodes['node_impact'] = nodes.w_from.fillna(0) + nodes.w_to.fillna(0)
    nodes = nodes.drop(columns=['w_from', 'w_to'])
    nodes['current_degree'] = 0
    nodes['splex'] = nodes.index
    nodes = nodes.reset_index().rename(columns={"index":"node_number"})

    edges = df.copy()
    edges['w'] = edges['w'] * (1-(edges['e']*2))
    edges['e'] = 0
    edges.loc[edges['w'] < 0]

    return nodes, edges, s

def write_solution(constr_edges, instance, algorithm = "construction"):
    file="output/"+instance+ "_"+algorithm+".txt"

    final_solution = constr_edges.loc[((constr_edges["e"]==0)&(constr_edges["w"]<=0))|
                                ((constr_edges["e"]==1)&(constr_edges["w"]>0)), ["n1", "n2"]]
    f = open(file, "w") 
    f.write(instance+"\n")
    f.close()
    final_solution.to_csv(file, mode='a', index=False, header=False, sep = " ")
 
 
 
def is_splex(nodes, plex_number, s) -> bool | pd.DataFrame:
    splex = nodes.loc[nodes["splex"] == plex_number]
    min_degree = len(splex.index) - s
    problem_nodes = splex.loc[splex["current_degree"] < min_degree]
    
    if len(problem_nodes.index) == 0:
        return True
    else:
        return problem_nodes

def extract1node(nodes_solution, edges_solution, score, s, step="best", random_state= 42):
    "step should be best, first or random"
    best_score = float('inf')
    # shuffle them, otherwise for random we will always get the first node removed
    nodes_shuffled = nodes_solution.sample(frac=1, random_state= random_state)
    
    # need to search whole neighborhood
    for index, row in nodes_shuffled.iterrows():
        current_node = row["node_number"]
        plex_number = row["splex"]
        node_edges = edges_solution.loc[((edges_solution["n1"]==current_node)|
                                         (edges_solution["n2"]==current_node)) &
                               (edges_solution["e"]==1)]
        node_neighbors = list(set(node_edges["n1"]).union(set(node_edges["n2"])))
        node_neighbors.remove(current_node)
        
        # what would node and edge list look like if we removed the node
        tmp_nodes = nodes_solution.copy()
        tmp_edges = edges_solution.copy()
        tmp_score = score - node_edges["w"].sum()
        tmp_score = tmp_score.item()

        # correct info of node we want to extract
        tmp_nodes.loc[tmp_nodes["node_number"]==current_node, ["current_degree"]] = 0
        tmp_nodes.loc[tmp_nodes["node_number"]==current_node, ["splex"]] = max(tmp_nodes.splex)+1
        # correct info of rest of nodes of this plex
        tmp_nodes.loc[tmp_nodes["node_number"].isin(node_neighbors), ["current_degree"]] -=1
        for i in node_neighbors:
            # remove the edge
            tmp_edges.loc[((tmp_edges["n1"]==current_node)&(tmp_edges["n2"]==i))|
                         ((tmp_edges["n2"]==current_node)&(tmp_edges["n1"]==i)), "e"] = 0
            weight = tmp_edges.loc[((tmp_edges["n1"]==current_node)&(tmp_edges["n2"]==i))|
                         ((tmp_edges["n2"]==current_node)&(tmp_edges["n1"]==i)), "w"]
            # adjust node impact in of both nodes
            tmp_nodes.loc[(tmp_nodes["node_number"]==i) | (tmp_nodes["node_number"]==current_node),
                         ["node_impact"]] -= weight 
            
        #check if it is still an splex
        plex = is_splex(tmp_nodes, plex_number = plex_number, s = s)
        if not isinstance(plex, (bool)): # we have to correct the splex
            corrected_nodes = set()
            potential_plex_nodes = list(plex.node_number.loc)
            for i in plex["node_number"]:
                if i in corrected_nodes:
                    continue # was already corrected when correcting another node
                else:
                    # get the cheapest edge we can add within the plex and add it
                    cheapest_edge = edges.loc[((tmp_edges["n1"]==i) & (tmp_edges["n2"].isin(potential_plex_nodes)) |
                                                  (tmp_edges["n2"]==i) & (tmp_edges["n1"].isin(potential_plex_nodes))) &
                                                 (tmp_edges["e"]==0)].sort_values(by=['w']).iloc[:1]
                    n1 = cheapest_edge.n1.item()
                    n2 = cheapest_edge.n2.item()
                    # add the cheapest edge
                    tmp_edges.loc[(tmp_edges["n1"]==n1)&(tmp_edges["n2"]==n2),
                                 "e"] = 1
                    # adjust node info
                    tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | 
                                  (tmp_nodes["node_number"]==n2),
                         ["current_degree"]] += 1
                    weight = tmp_edges.loc[(tmp_edges["n1"]==n1)&(tmp_edges["n2"]==n2),
                                 "w"].value
                    tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | 
                                  (tmp_nodes["node_number"]==n2),
                         ["node_impact"]] += weight
                    # update score
                    tmp_score += weight
           
                           
        # at this point we have a valid neighbor. Now we need to check if it is better
        if tmp_score <= best_score:
            print("found new best score")
            best_score = tmp_score
            new_nodes = tmp_nodes
            new_edges = tmp_edges
            
            if (step == "random"):             
                break
            if ((step == "first") and (best_score < score.item())):   
                break
                
    return (new_nodes, new_edges, best_score)


def merge2plexes(nodes_solution, edges_solution, score, s, step="best", random_state= 42):
    "step should be best, first or random"
    best_score = float('inf')
    plex_list = nodes_solution["splex"].unique()
    break_flag = False
    
    for plex_outer in plex_list:
        if break_flag == True:
            break # need to break out of outer loop too, if we want random or found a first best score
        for plex_inner in plex_list:
            if plex_inner < plex_outer:
                continue # we have already checked this combination
            else:
                tmp_nodes = nodes_solution.copy()
                tmp_edges = edges_solution.copy()
                tmp_score = score.item()
                # merge them
                tmp_nodes.loc[tmp_nodes["splex"]==plex_inner, ["splex"]] = plex_outer
                deg_needed = tmp_nodes.loc[tmp_nodes["splex"]==plex_outer].shape[0]-s
                potential_plex_nodes = list(tmp_nodes.node_number.loc[tmp_nodes["splex"]==plex_outer])
                
                #check if it is still an splex (most definitely not)
                plex = is_splex(tmp_nodes, plex_number = plex_outer, s = s)
                if not isinstance(plex, (bool)): # we have to correct the splex
                    corrected_nodes = set()                      
                    for i in plex["node_number"]: # for all problem nodes while they still need higher node degree
                        while np.any(tmp_nodes.loc[tmp_nodes["node_number"]==i, ["current_degree"]] < deg_needed):
                            # get the cheapest edge we can add within the plex
                            cheapest_edge = tmp_edges.loc[(((tmp_edges["n1"]==i) & (tmp_edges["n2"].isin(potential_plex_nodes))) |
                                                          ((tmp_edges["n2"]==i) & (tmp_edges["n1"].isin(potential_plex_nodes)))) &
                                                         (tmp_edges["e"]==0)].sort_values(by=['w']).iloc[:1]
                            n1 = cheapest_edge.n1.item()
                            n2 = cheapest_edge.n2.item()

                            # add the cheapest edge
                            tmp_edges.loc[((tmp_edges["n1"]==n1)&(tmp_edges["n2"]==n2)),"e"] = 1
                            # adjust node info
                            tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | (tmp_nodes["node_number"]==n2),
                                 ["current_degree"]] += 1
                            weight = tmp_edges.loc[(tmp_edges["n1"]==n1) & (tmp_edges["n2"]==n2),
                                         "w"].item()
                            tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | (tmp_nodes["node_number"]==n2),
                                 ["node_impact"]] += weight
                            # update score
                            tmp_score += weight
           
                # at this point we have a valid neighbor. Now we need to check if it is better
                if tmp_score <= best_score:
                    print("found new best score")
                    best_score = tmp_score
                    new_nodes = tmp_nodes
                    new_edges = tmp_edges

                    if (step == "random"):
                        break_flag = True
                        break
                    if ((step == "first") and (best_score < score.item())): 
                        break_flag = True
                        break
                
    return (new_nodes, new_edges, best_score)