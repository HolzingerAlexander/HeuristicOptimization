import numpy as np
import pandas as pd
import time
import random
import math

def create_problem_instance(path):
    metadata = pd.read_csv(path, sep=" ", nrows=1, header=None).iloc[0]
    s = metadata.iloc[0]
    n = metadata.iloc[1]
    m = metadata.iloc[2]

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

    return nodes, edges, s, n, m

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
    nodes_shuffled = nodes_solution
    if step == "random":
        nodes_shuffled = nodes_shuffled.sample(frac=1, random_state= random_state)
    start = time.time()
    
    # need to search whole neighborhood
    for _, row in nodes_shuffled.iterrows():
        current_node = row["node_number"]
        plex_number = row["splex"]
        node_edges = edges_solution.loc[((edges_solution["n1"]==current_node)|
                                         (edges_solution["n2"]==current_node)) &
                               (edges_solution["e"]==1)]
        node_neighbors = list(set(node_edges["n1"]).union(set(node_edges["n2"])))
        if node_neighbors: # if this node didn't form its own splex
            node_neighbors.remove(current_node)
        
        # what would node and edge list look like if we removed the node
        tmp_score = score - node_edges["w"].sum().item()

        # The score will not change. Removing one node will always leave a valid s-Plex
        if tmp_score > best_score:
            continue

        tmp_nodes = nodes_solution.copy()
        tmp_edges = edges_solution.copy()

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
                           
        # at this point we have a valid neighbor. Now we need to check if it is better
        if tmp_score <= best_score:
            #print("found new best score")
            best_score = tmp_score
            new_nodes = tmp_nodes
            new_edges = tmp_edges
            
            if (step == "random"):             
                break
            if ((step == "first") and (best_score < score)):   
                break
    
    print(round(time.time()-start, 2), "seconds")
    return (new_nodes, new_edges, best_score)


def merge2plexes(nodes_solution, edges_solution, score, s, step="best", random_state= 42, add_neg_edges = False):
    "step should be best, first or random"
    best_score = float('inf')
    plex_list = nodes_solution["splex"].unique()
    break_flag = False
    start = time.time()
    
    for plex_outer in plex_list:
        if break_flag == True:
            break # need to break out of outer loop too, if we want random or found a first best score
        for plex_inner in plex_list:
            if plex_inner <= plex_outer:
                continue # we have already checked this combination
            else:
                tmp_nodes = nodes_solution.copy()
                tmp_edges = edges_solution.copy()
                tmp_score = score
                # merge them
                tmp_nodes.loc[tmp_nodes["splex"]==plex_inner, ["splex"]] = plex_outer
                deg_needed = tmp_nodes.loc[tmp_nodes["splex"]==plex_outer].shape[0]-s
                potential_plex_nodes = list(tmp_nodes.node_number.loc[tmp_nodes["splex"]==plex_outer])
                # add all edges we want to have anyway
                if add_neg_edges:
                    for index, row in tmp_edges.iterrows():
                        n1 = row.n1.item()
                        n2 = row.n2.item()
                        if (n1 in (potential_plex_nodes)) and (n2 in (potential_plex_nodes)) and row.e == 0 and row.w <=0:

                            tmp_edges.loc[(tmp_edges["n1"]==n1)&(tmp_edges["n2"]==n2), "e"] = 1
                            # update node info
                            tmp_nodes.loc[(tmp_nodes["node_number"]==n1)|(tmp_nodes["node_number"]==n2), 
                                          ["current_degree"]] +=1
                            tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | (tmp_nodes["node_number"]==n2),
                                     ["node_impact"]] += row.w 
                            # update score
                            tmp_score += row.w
                        
                #check if it is still an splex (most definitely not)
                plex = is_splex(tmp_nodes, plex_number = plex_outer, s = s)
                if not isinstance(plex, (bool)): # we have to correct the splex
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
                    #print("found new best score")
                    best_score = tmp_score
                    new_nodes = tmp_nodes
                    new_edges = tmp_edges

                    if (step == "random"):
                        break_flag = True
                        break
                    if ((step == "first") and (best_score < score)): 
                        break_flag = True
                        break
    
    print(round(time.time()-start, 2), "seconds")            
    return (new_nodes, new_edges, best_score)

def construction_heuristic(nodes_orig, edges_orig, s, number_of_exisisting_edges):
    nodes = nodes_orig.copy()
    edges = edges_orig.copy()
    
    start = time.time()
    
    existing_edges = edges.loc[edges["w"]<0].sort_values("w")
    iteration = 0
    last_success = 0

    print("start")

    for _, row in existing_edges.iterrows():
        # check if this edge was already added, then we can continue to next one
        if edges.loc[(edges["n1"]==row["n1"]) & (edges["n2"]==row["n2"]), "e"].values[0] == 1:
            #print(iteration)
            iteration += 1
            continue
        
        if iteration - last_success > 2000: #((number_of_exisisting_edges - added_edges) / 2):
            break

        # get plex assignment of both nodes
        n1_plex = nodes.loc[nodes["node_number"]==row["n1"], "splex"].values[0]
        n2_plex = nodes.loc[nodes["node_number"]==row["n2"], "splex"].values[0]
        # all nodes that would be in the plex if we merged it
        nodes_in_plex = nodes.loc[(nodes["splex"]==n1_plex) | (nodes["splex"] ==n2_plex), "node_number"].values
        # all edges we would want anyway as they were in original assignment
        edges_to_add = edges.loc[(edges["n1"].isin(nodes_in_plex)) & 
                                 (edges["n2"].isin(nodes_in_plex)) & # only edges within the potential splex
                                 (edges["w"]<=0) & # that we want to add anyway
                                 (edges["e"]==0)] #that have not been added yet

        number_of_nodes = len(nodes_in_plex)
        edges_missing = 0
        needed_edges_by_node: dict = dict()

        # check if it would be a valid splex if we merge by checking node degrees
        for node in nodes_in_plex:
            node_degree = nodes.loc[nodes["node_number"]==node,"current_degree"].values[0]

            if node_degree < (number_of_nodes - s):
                # if we can reach the node degree by only using the edges we want to add anyway, then everything is fine
                num_potential_edges = edges_to_add.loc[(edges_to_add["n1"]==node)|
                                                             (edges_to_add["n2"]==node)].shape[0]
                new_degree = node_degree + num_potential_edges
                needed_edges = number_of_nodes - s - new_degree

                if needed_edges > 0:
                    edges_missing += needed_edges
                    needed_edges_by_node[node] = needed_edges

        totalWeight = edges_to_add["w"].sum()

        if edges_missing >= (totalWeight/(-2)):
            continue

        if edges_missing:
            nodes_to_process = needed_edges_by_node.keys()
            
            candidates = edges.loc[(edges["n1"].isin(nodes_in_plex)) & 
                                    (edges["n2"].isin(nodes_in_plex)) & # only edges within the potential splex
                                    (edges["w"]>0) & # that we do not add automatically
                                    (edges["e"]==0) & # that have not been added yet
                                    (edges["n1"].isin(nodes_to_process) |
                                    edges["n2"].isin(nodes_to_process))] # that have at least one node with missing edges
            candidates = candidates.sort_values(by=['w'])

            bad_edges_to_add = pd.DataFrame(columns=["n1","n2","e","w"])

            while needed_edges_by_node and (not candidates.empty) and (totalWeight < 0):
                current = candidates.iloc[0]
                candidates = candidates.iloc[1:]

                if (not (current["n1"] in needed_edges_by_node)) \
                   and (not (current["n2"] in needed_edges_by_node)):
                   continue

                bad_edges_to_add = pd.concat([bad_edges_to_add, current.to_frame().T])
                edge_weight = current["w"]

                node1 = current["n1"]
                if node1 in needed_edges_by_node:
                    needed_edges_by_node[node1] -= 1
                    if needed_edges_by_node[node1] == 0:
                        needed_edges_by_node.pop(node1)
                        
                node2 = current["n2"]
                if node2 in needed_edges_by_node:
                    needed_edges_by_node[node2] -= 1
                    if needed_edges_by_node[node2] == 0:
                        needed_edges_by_node.pop(node2)

                totalWeight += edge_weight
                #print ("totalWeight: " + str(totalWeight))
                #print ("candidates: " + str(len(candidates.index)))

            if not needed_edges_by_node and totalWeight < 0:
                edges_missing = 0
                edges_to_add = pd.concat([edges_to_add, bad_edges_to_add])

        # if it would be a valid splex, actually merge it
        if not edges_missing:
            # merge them
            nodes.loc[nodes["splex"]==n2_plex, "splex"] = n1_plex
            # include all edges we want to add
            for index, oedge in edges_to_add.iterrows():
                edges.loc[(edges["n1"]==oedge["n1"]) & (edges["n2"]==oedge["n2"]), "e"] = 1
                # update node info (degree and node impact)
                nodes.loc[nodes["node_number"]==oedge["n1"], "current_degree"] +=1
                nodes.loc[nodes["node_number"]==oedge["n2"], "current_degree"] +=1
                nodes.loc[nodes["node_number"]==oedge["n1"], "node_impact"] -= abs(oedge["w"])
                nodes.loc[nodes["node_number"]==oedge["n2"], "node_impact"] -= abs(oedge["w"])

            last_success = iteration

        #print(iteration)
        iteration += 1
            
    print(round(time.time()-start, 2), "seconds")
    score = edges.loc[((edges["e"]==0)&(edges["w"]<=0))| 
                        ((edges["e"]==1)&(edges["w"]>0)), ["w"]].abs().sum()
    return (nodes, edges, score.item())

def randomized_greedy(nodes_orig, edges_orig, s, alpha=0.5, random_seed = None):
    nodes = nodes_orig.copy()
    edges = edges_orig.copy()
    
    if random_seed is not None:
        random.seed(random_seed)
    start = time.time()
    
    existing_edges = edges.loc[edges["w"]<0].sort_values("w") # this is our candidate list
    
    iteration = 0
    last_success = 0

    while existing_edges.shape[0]>0:
        #build restricted candidate list
        costs_threshold = max(existing_edges["w"]) + alpha * (min(existing_edges["w"]) - max(existing_edges["w"]))
        rcl = existing_edges.loc[existing_edges["w"]>= costs_threshold]
        # pick one edge at random
        row = rcl.sample(1)

        if edges.loc[(edges["n1"]==row["n1"].values[0]) & (edges["n2"]==row["n2"].values[0]), "e"].values[0] == 1:
            print(iteration)
            iteration += 1
            continue
        
        if iteration - last_success > 1000: #((number_of_exisisting_edges - added_edges) / 2):
            break
            
        # get plex assignment of both nodes
        n1_plex = nodes.loc[nodes["node_number"]==row["n1"].values[0], "splex"].values[0]
        n2_plex = nodes.loc[nodes["node_number"]==row["n2"].values[0], "splex"].values[0]
        # all nodes that would be in the plex if we merged it
        nodes_in_plex = nodes.loc[(nodes["splex"]==n1_plex) | (nodes["splex"] ==n2_plex), "node_number"].values
        # all edges we would want anyway as they were in original assignment
        edges_to_add = edges.loc[(edges["n1"].isin(nodes_in_plex)) & 
                                        (edges["n2"].isin(nodes_in_plex)) & # only edges within the potential splex
                                       (edges["w"]<=0) & # that we want to add anyway
                                       (edges["e"]==0)] #that have not been added yet

        number_of_nodes = len(nodes_in_plex)
        edges_missing = False
        needed_edges_by_node = dict()

        # check if it would be a valid splex if we merge by checking node degrees
        for node in nodes_in_plex:
            node_degree = nodes.loc[nodes["node_number"]==node,"current_degree"].values[0]

            if node_degree < (number_of_nodes - s):
                # if we can reach the node degree by only using the edges we want to add anyway, then everything is fine
                num_potential_edges = edges_to_add.loc[(edges_to_add["n1"]==node)|
                                                             (edges_to_add["n2"]==node)].shape[0]
                new_degree = node_degree + num_potential_edges
                needed_edges = number_of_nodes - s - new_degree

                if needed_edges > 0:
                    edges_missing += needed_edges
                    needed_edges_by_node[node] = needed_edges

        totalWeight = edges_to_add["w"].sum()

        if edges_missing >= (totalWeight/(-2)):
            continue

        if edges_missing:
            nodes_to_process = needed_edges_by_node.keys()
            
            candidates = edges.loc[(edges["n1"].isin(nodes_in_plex)) & 
                                    (edges["n2"].isin(nodes_in_plex)) & # only edges within the potential splex
                                    (edges["w"]>0) & # that we do not add automatically
                                    (edges["e"]==0) & # that have not been added yet
                                    (edges["n1"].isin(nodes_to_process) |
                                    edges["n2"].isin(nodes_to_process))] # that have at least one node with missing edges
            candidates = candidates.sort_values(by=['w'])

            bad_edges_to_add = pd.DataFrame(columns=["n1","n2","e","w"])

            while needed_edges_by_node and (not candidates.empty) and (totalWeight < 0):
                current = candidates.iloc[0]
                candidates = candidates.iloc[1:]

                if (not (current["n1"] in needed_edges_by_node)) \
                   and (not (current["n2"] in needed_edges_by_node)):
                   continue

                bad_edges_to_add = pd.concat([bad_edges_to_add, current.to_frame().T])
                edge_weight = current["w"]

                node1 = current["n1"]
                if node1 in needed_edges_by_node:
                    needed_edges_by_node[node1] -= 1
                    if needed_edges_by_node[node1] == 0:
                        needed_edges_by_node.pop(node1)
                        
                node2 = current["n2"]
                if node2 in needed_edges_by_node:
                    needed_edges_by_node[node2] -= 1
                    if needed_edges_by_node[node2] == 0:
                        needed_edges_by_node.pop(node2)

                totalWeight += edge_weight
                #print ("totalWeight: " + str(totalWeight))
                #print ("candidates: " + str(len(candidates.index)))

            if not needed_edges_by_node and totalWeight < 0:
                edges_missing = 0
                edges_to_add = pd.concat([edges_to_add, bad_edges_to_add])

        # if it would be a valid splex, actually merge it
        if not edges_missing:
            # merge them
            nodes.loc[nodes["splex"]==n2_plex, "splex"] = n1_plex
            # include all edges we want to add
            for index, oedge in edges_to_add.iterrows():
                edges.loc[(edges["n1"]==oedge["n1"]) & (edges["n2"]==oedge["n2"]), "e"] = 1
                # update node info (degree and node impact)
                nodes.loc[nodes["node_number"]==oedge["n1"], "current_degree"] +=1
                nodes.loc[nodes["node_number"]==oedge["n2"], "current_degree"] +=1
                nodes.loc[nodes["node_number"]==oedge["n1"], "node_impact"] -= abs(oedge["w"])
                nodes.loc[nodes["node_number"]==oedge["n2"], "node_impact"] -= abs(oedge["w"])
            last_success = iteration

        iteration += 1

        # remove the edge from the candidate list
        existing_edges = existing_edges.drop(existing_edges[(existing_edges.n1 == row.n1.values[0])&
                                                            (existing_edges.n2 == row.n2.values[0])].index)
    score = edges.loc[((edges["e"]==0)&(edges["w"]<=0))| 
                        ((edges["e"]==1)&(edges["w"]>0)), ["w"]].abs().sum()    
    print(round(time.time()-start, 2), "seconds")
    return(nodes, edges, score.item())

def remove_edge(nodes_solution, edges_solution, score, s, step="best", random_state= 42, add_neg_edges = False, skip_invalid = False):
    "step should be best, first or random"
    best_score = score
    existing_edges = edges_solution.loc[(edges_solution["e"]==1) & (edges_solution["w"] > 0)]
    if step == "random":
        existing_edges = existing_edges.sample(frac=1, random_state= random_state) 
    start = time.time()
    
    new_nodes = nodes_solution
    new_edges = edges_solution

    for _, edge in existing_edges.iterrows(): # randomize this
        # get the plex number
        plex_number = nodes_solution.splex.loc[nodes_solution["node_number"]==edge["n1"]].item()
        edge_nodes = nodes_solution.loc[(nodes_solution["node_number"]==edge["n1"])
                                        |(nodes_solution["node_number"]==edge["n2"])]
        plex_nodes = nodes_solution.loc[nodes_solution["splex"] == plex_number]

        min_degree = len(plex_nodes.index) - s
        if skip_invalid:
            if len(edge_nodes.loc[edge_nodes["current_degree"] < (min_degree + 1)].index) > 0:
                continue
            if edge.w < 0:
                continue

        tmp_edges = edges_solution.copy()
        tmp_nodes = nodes_solution.copy()
        tmp_score = score

        # remove the edge
        tmp_edges.loc[(tmp_edges["n1"]==edge["n1"])&(tmp_edges["n2"]==edge["n2"]), "e"] = 0
        #update node infos and score
        tmp_nodes.loc[(tmp_nodes["node_number"]==edge["n1"])|(tmp_nodes["node_number"]==edge["n2"]), 
                                          ["current_degree"]] -=1
        tmp_nodes.loc[(tmp_nodes["node_number"]==edge["n1"]) | (tmp_nodes["node_number"]==edge["n2"]),
                                     ["node_impact"]] -= edge.w 
        # update score
        tmp_score -= edge.w
        
        #check if it is still an splex
        plex = is_splex(tmp_nodes, plex_number = plex_number, s = s)
        if not isinstance(plex, (bool)): # we have to correct the splex
            print("b")
            potential_plex_nodes = list(tmp_nodes.node_number.loc[tmp_nodes["splex"]==plex_number])
            for i in plex["node_number"]: # should return 2 nodes at most
                deg_needed = tmp_nodes.loc[tmp_nodes["splex"]==plex_number].shape[0]-s
                while np.any(tmp_nodes.loc[tmp_nodes["node_number"]==i, ["current_degree"]] < deg_needed):
                    # get all edge we can add within the plex
                    potential_edges = tmp_edges.loc[(((tmp_edges["n1"]==i) & (tmp_edges["n2"].isin(potential_plex_nodes))) |
                                                          ((tmp_edges["n2"]==i) & (tmp_edges["n1"].isin(potential_plex_nodes)))) &
                                                         (tmp_edges["e"]==0)]
                    # but exclude the one we just removed
                    potential_edges = potential_edges.loc[~((potential_edges["n1"]==edge["n1"])&
                                                            (potential_edges["n2"]==edge["n2"]))]
                    # get the cheapest
                    cheapest_edge = potential_edges.sort_values(by=['w']).iloc[:1]
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
            #print("found new best score")
            best_score = tmp_score
            new_nodes = tmp_nodes
            new_edges = tmp_edges

            if (step == "random"):
                break
            if ((step == "first") and (best_score < score)): 
                break
    
    print(round(time.time()-start, 2), "seconds")            
    return (new_nodes, new_edges, best_score)

def add_edge_within_plex(nodes_solution, edges_solution, score, s, step="best", random_state= 42):
    "step should be best, first or random"
    best_score = float('inf')
    random.seed(random_state)
    plexes = list(nodes_solution["splex"].unique())
    if (step == "random"):
        plexes = random.sample(plexes, len(plexes))
    break_flag = False
    start = time.time()

    new_nodes = nodes_solution
    new_edges = edges_solution
    
    for plex in plexes:
        if break_flag:
            break
        plex_nodes = nodes_solution.node_number.loc[nodes_solution["splex"]==plex]
        potential_edges = edges_solution.loc[(edges_solution["n1"].isin(plex_nodes))& (edges_solution["n2"].isin(plex_nodes)) &
                                            (edges_solution["e"]==0)] 
        if (step == "random"):
            potential_edges = potential_edges.sample(frac=1, random_state= random_state) #shuffle them

        for _, edge in potential_edges.iterrows():
            tmp_score = score + edge.w
            
            # at this point we have a valid neighbor. Now we need to check if it is better
            if tmp_score <= best_score:
                #print("found new best score")

                tmp_edges = edges_solution.copy()
                tmp_nodes = nodes_solution.copy()

                # add the edge
                tmp_edges.loc[(tmp_edges["n1"]==edge["n1"])&(tmp_edges["n2"]==edge["n2"]), "e"] = 1
                #update node infos and score
                tmp_nodes.loc[(tmp_nodes["node_number"]==edge["n1"])|(tmp_nodes["node_number"]==edge["n2"]), 
                                            ["current_degree"]] +=1
                tmp_nodes.loc[(tmp_nodes["node_number"]==edge["n1"]) | (tmp_nodes["node_number"]==edge["n2"]),
                                        ["node_impact"]] += edge.w 

                best_score = tmp_score
                new_nodes = tmp_nodes
                new_edges = tmp_edges

                if (step == "random"):
                    break_flag = True
                    break
                if ((step == "first") and (best_score < score)): 
                    break_flag = True
                    break
    
    print(round(time.time()-start, 2), "seconds")            
    return (new_nodes, new_edges, best_score)


class Solution:
    def __init__(self, nodes, edges, s, weight):
        self.nodes = nodes
        self.edges = edges
        self.s = s
        self.weight = weight

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        return self.edges
    
    def get_s(self):
        return self.s

    def get_weight(self):
        return self.weight

class Problem:
    def __init__(self, path):
        self.nodes, self.edges, self.s, self.n, self.m = create_problem_instance(path)

    def get_heuristic_solution(self) -> Solution:
        sol_nod, sol_edg, sol_weight = construction_heuristic(self.nodes, self.edges, self.s, self.m)
        return Solution(sol_nod, sol_edg, self.s, sol_weight)

    def get_randomized_solution(self) -> Solution:
        sol_nod, sol_edg, sol_weight = randomized_greedy(self.nodes, self.edges, self.s)
        return Solution(sol_nod, sol_edg, self.s, sol_weight)

class Neighborhood: 
    def __init__(self, improvement_fun):
        self.improvement_fun = improvement_fun

    def get_improvement(self, solution: Solution, improvement_type: str):
        nodes, edges, weight = self.improvement_fun(solution, improvement_type)
        return Solution(nodes, edges, solution.get_s(), weight)

from enum import Enum

class ImprovementType(Enum):
    FIRST = 1
    BEST = 2
    RANDOM = 3

def local_search(solution: Solution, neighborhood: Neighborhood, improve: ImprovementType, times: int = 20) -> Solution:
    current_solution: Solution = solution
    for _ in range(times):
        neighbor: Solution
        match improve:
            case ImprovementType.FIRST:
                neighbor = neighborhood.get_improvement(current_solution, "first")
            case ImprovementType.BEST:
                neighbor = neighborhood.get_improvement(current_solution, "best")
            case ImprovementType.RANDOM:
                neighbor = neighborhood.get_improvement(current_solution, "random")
        
        if neighbor.get_weight() < current_solution.get_weight():
            current_solution =  neighbor
        elif improve == ImprovementType.BEST:
            break
    
    return current_solution

def vnd(solution: Solution, neighborhoods: list[Neighborhood], max_minutes = 15) -> Solution:
    current_best: Solution = solution
    
    start = time.time()

    i = 0 # First neighborhood
    while i < len(neighborhoods):
        current_neighborhood: Neighborhood = neighborhoods[i]
        local_optimum = current_neighborhood.get_improvement(current_best, "first")

        print("i: " + str(i) + ", " + str(local_optimum.get_weight()))
        if not valid(local_optimum):
            break
            
        if not consistent(local_optimum):
            break

        if local_optimum.get_weight() < current_best.get_weight():
            current_best = local_optimum
            i = 0
        else:
            i += 1
        print((time.time()-start))
        if (time.time()-start) > (max_minutes * 60):
            break

    return current_best

def grasp(problem: Problem, neighborhoods: list[Neighborhood], times: int) -> Solution:
    best_solution: Solution = None
    for i in range(times):
        current_solution = problem.get_randomized_solution()
        local_optimum = vnd(current_solution, neighborhoods, 15)
        if (best_solution == None) or (local_optimum.get_weight() < best_solution.get_weight()):
            best_solution = local_optimum
    return best_solution

def move1node(nodes_solution, edges_solution, score, s, step="best", random_state= 42, add_neg_edges = True, print_info = True):
    "step should be best, first or random"
    best_score = float('inf')
    break_flag = False
    # shuffle them, otherwise for random we will always get the first node removed
    nodes_shuffled = nodes_solution.sample(frac=1, random_state= random_state)
    start = time.time()
    
    # need to search whole neighborhood
    for _, row in nodes_shuffled.iterrows(): #loop through nodes
        if break_flag:
            break
        current_node = row["node_number"]
        plex_number = row["splex"]
        node_edges = edges_solution.loc[((edges_solution["n1"]==current_node)|
                                         (edges_solution["n2"]==current_node)) &
                               (edges_solution["e"]==1)]
        
        # what would node and edge list look like if we removed the node
        tmp_nodes = nodes_solution.copy()
        tmp_edges = edges_solution.copy()
        tmp_score = score - node_edges["w"].sum()
        tmp_score = tmp_score.item()
        
        node_neighbors = list(set(node_edges["n1"]).union(set(node_edges["n2"])))
        if node_neighbors: # if this node didn't form its own splex
            node_neighbors.remove(current_node)

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
                         ((tmp_edges["n2"]==current_node)&(tmp_edges["n1"]==i)), "w"].item()
            # adjust node impact of both nodes
            tmp_nodes.loc[(tmp_nodes["node_number"]==i) | (tmp_nodes["node_number"]==current_node),
                         ["node_impact"]] -= weight
            
        #check if it is still an splex and correct if necessary
        if node_neighbors: # only need to do this if the node was not alone
            plex = is_splex(tmp_nodes, plex_number = plex_number, s = s)
            if not isinstance(plex, (bool)): # we have to correct the splex
                corrected_nodes = set()
                potential_plex_nodes = list(tmp_nodes.node_number.loc[tmp_nodes["splex"]==plex_number])
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
                                     "w"].item()
                        tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | 
                                      (tmp_nodes["node_number"]==n2),
                             ["node_impact"]] += weight
                        # update score
                        tmp_score += weight
            
            # check if we can remove bad edges without breaking the s-plex condition
            deg_needed = tmp_nodes.loc[tmp_nodes["splex"]==plex_number].shape[0] -s
            high_deg = list(tmp_nodes.loc[(tmp_nodes["splex"]==plex_number)&(tmp_nodes["current_degree"]>deg_needed), "node_number"])
            # get the "bad" existing edges of all nodes with too high degree
            potential_edges = tmp_edges.loc[(tmp_edges["n1"].isin(high_deg)) & (tmp_edges["n2"].isin(high_deg)) & 
                                            (tmp_edges["e"]==1) &
                                           (tmp_edges["w"]>0)].sort_values(by=["w"], ascending = False)
            for _, row in potential_edges.iterrows():
                n1 = row.n1.item()
                n2 = row.n2.item()
                w = row.w.item()
                if not high_deg: 
                    break
                    # as long as there are still edges with too high degree
                if (n1 in high_deg) and (n2 in high_deg):
                    # we can remove this edge
                    tmp_edges.loc[(tmp_edges["n1"]==n1)&(tmp_edges["n2"]==n2), "e"]=0
                    #update nodes and score
                    tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | 
                                  (tmp_nodes["node_number"]==n2),
                                        ["current_degree"]] -= 1
                    tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | 
                                      (tmp_nodes["node_number"]==n2),
                                    ["node_impact"]] -= w
                    tmp_score -= w
                        
                    # check if the degrees are still too high
                    if tmp_nodes.loc[(tmp_nodes["node_number"]==n1), "current_degree"].item() == deg_needed:
                        high_deg.remove(n1)
                    if tmp_nodes.loc[(tmp_nodes["node_number"]==n2), "current_degree"].item() == deg_needed:
                        high_deg.remove(n2)

                
        # now we can see to which other plex to move it
        # we shuffle all the plex numbers we currently have. This way, node can stay in the one we jsut created and stay
        # its own plex if it is better to do it like that.
        plexes_shuffled = random.sample(list(tmp_nodes["splex"].unique()), len(tmp_nodes["splex"].unique()))
        for plex_to in plexes_shuffled:
            if plex_to == plex_number:
                continue #this is the plex we just removed the node from
            else:
                tmp_nodes_inner = tmp_nodes.copy()
                tmp_edges_inner = tmp_edges.copy()
                tmp_score_inner = tmp_score
                # merge them
                #tmp_nodes_inner.loc[tmp_nodes_inner["splex"]==plex_to, ["splex"]] = plex_to
                tmp_nodes_inner.loc[tmp_nodes_inner["node_number"]==current_node, ["splex"]] = plex_to
                deg_needed = tmp_nodes_inner.loc[tmp_nodes_inner["splex"]==plex_to].shape[0]-s
                potential_plex_nodes = list(tmp_nodes_inner.node_number.loc[tmp_nodes_inner["splex"]==plex_to])
                # add all edges we want to have anyway
                if add_neg_edges:
                    for index, row2 in tmp_edges_inner.iterrows():
                        n1 = row2.n1.item()
                        n2 = row2.n2.item()
                        w = row2.w.item()
                        if (n1 in (potential_plex_nodes)) and (n2 in (potential_plex_nodes)) and row2.e == 0 and row2.w <=0:

                            tmp_edges_inner.loc[(tmp_edges_inner["n1"]==n1)&(tmp_edges_inner["n2"]==n2), "e"] = 1
                            # update node info
                            tmp_nodes_inner.loc[(tmp_nodes_inner["node_number"]==n1)|(tmp_nodes_inner["node_number"]==n2), 
                                          ["current_degree"]] +=1
                            tmp_nodes_inner.loc[(tmp_nodes_inner["node_number"]==n1) | (tmp_nodes_inner["node_number"]==n2),
                                     ["node_impact"]] += w 
                            # update score
                            tmp_score_inner += w
                        
                #check if it is still an splex
                plex = is_splex(tmp_nodes_inner, plex_number = plex_to, s = s)
                if not isinstance(plex, (bool)): # we have to correct the splex
                    corrected_nodes = set()
                    for i in list(plex["node_number"]): # for all problem nodes while they still need higher node degree
                        while tmp_nodes_inner.loc[tmp_nodes_inner["node_number"]==i, "current_degree"].item() < deg_needed:
                            # get the cheapest edge we can add within the plex
                            potential_edges = tmp_edges_inner.loc[(((tmp_edges_inner["n1"]==i) & 
                                                                  (tmp_edges_inner["n2"].isin(potential_plex_nodes))) |
                                                                 ((tmp_edges_inner["n2"]==i) & 
                                                                  (tmp_edges_inner["n1"].isin(potential_plex_nodes)))) &
                                                         (tmp_edges_inner["e"]==0)].sort_values(by=['w'])

                            cheapest_edge = potential_edges.iloc[:1]
                            n1 = cheapest_edge.n1.item()
                            n2 = cheapest_edge.n2.item()

                            # add the cheapest edge
                            tmp_edges_inner.loc[((tmp_edges_inner["n1"]==n1)&(tmp_edges_inner["n2"]==n2)),"e"] = 1
                            # adjust node info
                            tmp_nodes_inner.loc[(tmp_nodes_inner["node_number"]==n1) | (tmp_nodes_inner["node_number"]==n2),
                                 ["current_degree"]] += 1
                            weight = tmp_edges_inner.loc[(tmp_edges_inner["n1"]==n1) & (tmp_edges_inner["n2"]==n2),
                                         "w"].item()
                            tmp_nodes_inner.loc[(tmp_nodes["node_number"]==n1) | (tmp_nodes_inner["node_number"]==n2),
                                 ["node_impact"]] += weight
                            # update score
                            tmp_score_inner += weight     
                           
                # at this point we have a valid neighbor. Now we need to check if it is better
                if tmp_score_inner <= best_score:
                    #if print_info:
                        #print("found new best score")
                    best_score = tmp_score_inner
                    new_nodes = tmp_nodes_inner
                    new_edges = tmp_edges_inner

                    if (step == "random"):  
                        break_flag = True
                        break
                    if ((step == "first") and (best_score < score)):   
                        break_flag = True
                        break
    
    if print_info:
        print(round(time.time()-start, 2), "seconds")
    return (new_nodes, new_edges, best_score)    
    
def move1node_random(nodes_solution, edges_solution, score, s, add_neg_edges = True, print_info = True):
    "step should be best, first or random"
    # pick a random node to move
    row = nodes_solution.sample(1)#nodes_shuffled = nodes_solution.sample(frac=1, random_state= random_state)
    
    start = time.time()
    # get node infos
    current_node = row["node_number"].item()
    plex_number = row["splex"].item()
    node_edges = edges_solution.loc[((edges_solution["n1"]==current_node)|
                                     (edges_solution["n2"]==current_node)) &
                               (edges_solution["e"]==1)]
        
    # what would node and edge list look like if we removed the node
    tmp_nodes = nodes_solution.copy()
    tmp_edges = edges_solution.copy()
    tmp_score = score - node_edges["w"].sum()
    tmp_score = tmp_score.item()
        
    node_neighbors = list(set(node_edges["n1"]).union(set(node_edges["n2"])))
    if node_neighbors: # if this node didn't form its own splex
        node_neighbors.remove(current_node)

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
                         ((tmp_edges["n2"]==current_node)&(tmp_edges["n1"]==i)), "w"].item()
        # adjust node impact of both nodes
        tmp_nodes.loc[(tmp_nodes["node_number"]==i) | (tmp_nodes["node_number"]==current_node),
                         ["node_impact"]] -= weight
            
    #check if it is still an splex and correct if necessary
    if node_neighbors: # only need to do this if the node was not alone
        plex = is_splex(tmp_nodes, plex_number = plex_number, s = s)
        if not isinstance(plex, (bool)): # we have to correct the splex
            corrected_nodes = set()
            potential_plex_nodes = list(tmp_nodes.node_number.loc[tmp_nodes["splex"]==plex_number])
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
                                     "w"].item()
                    tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | 
                                      (tmp_nodes["node_number"]==n2),
                             ["node_impact"]] += weight
                    # update score
                    tmp_score += weight
            
        # check if we can remove bad edges without breaking the s-plex condition
        deg_needed = tmp_nodes.loc[tmp_nodes["splex"]==plex_number].shape[0] -s
        high_deg = list(tmp_nodes.loc[(tmp_nodes["splex"]==plex_number)&(tmp_nodes["current_degree"]>deg_needed), "node_number"])
        # get the "bad" existing edges of all nodes with too high degree
        potential_edges = tmp_edges.loc[(tmp_edges["n1"].isin(high_deg)) & (tmp_edges["n2"].isin(high_deg)) & 
                                            (tmp_edges["e"]==1) &
                                           (tmp_edges["w"]>0)].sort_values(by=["w"], ascending = False)
        for _, row2 in potential_edges.iterrows():
            n1 = row2.n1.item()
            n2 = row2.n2.item()
            w = row2.w.item()
            if not high_deg: 
                break
                # as long as there are still edges with too high degree
            if (n1 in high_deg) and (n2 in high_deg):
                # we can remove this edge
                tmp_edges.loc[(tmp_edges["n1"]==n1)&(tmp_edges["n2"]==n2), "e"]=0
                #update nodes and score
                tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | 
                                  (tmp_nodes["node_number"]==n2),
                                        ["current_degree"]] -= 1
                tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | 
                                      (tmp_nodes["node_number"]==n2),
                                    ["node_impact"]] -= w
                tmp_score -= w
                        
                # check if the degrees are still too high
                if tmp_nodes.loc[(tmp_nodes["node_number"]==n1), "current_degree"].item() == deg_needed:
                        high_deg.remove(n1)
                if tmp_nodes.loc[(tmp_nodes["node_number"]==n2), "current_degree"].item() == deg_needed:
                        high_deg.remove(n2)

    # now we can see to which other plex to move it
    plex_to = random.sample(list(tmp_nodes["splex"].unique()), 1)[0]
    # merge them
    tmp_nodes.loc[tmp_nodes["node_number"]==current_node, "splex"] = plex_to
    deg_needed = tmp_nodes.loc[tmp_nodes["splex"]==plex_to].shape[0]-s
    potential_plex_nodes = list(tmp_nodes.node_number.loc[tmp_nodes["splex"]==plex_to])
    # add all edges we want to have anyway
    if add_neg_edges:
        for index, row3 in tmp_edges.iterrows():
            n1 = row3.n1.item()
            n2 = row3.n2.item()
            w = row3.w.item()
            if (n1 in (potential_plex_nodes)) and (n2 in (potential_plex_nodes)) and row3.e == 0 and row3.w <=0:
                tmp_edges.loc[(tmp_edges["n1"]==n1)&(tmp_edges["n2"]==n2), "e"] = 1
                # update node info
                tmp_nodes.loc[(tmp_nodes["node_number"]==n1)|(tmp_nodes["node_number"]==n2), 
                                          ["current_degree"]] +=1
                tmp_nodes.loc[(tmp_nodes["node_number"]==n1) | (tmp_nodes["node_number"]==n2),
                                     ["node_impact"]] += w 
                # update score
                tmp_score += w
                        
    #check if it is still an splex
    plex = is_splex(tmp_nodes, plex_number = plex_to, s = s)
    if not isinstance(plex, (bool)): # we have to correct the splex
        corrected_nodes = set()
        for i in list(plex["node_number"]): # for all problem nodes while they still need higher node degree
            while tmp_nodes.loc[tmp_nodes["node_number"]==i, "current_degree"].item() < deg_needed:
                # get the cheapest edge we can add within the plex
                potential_edges = tmp_edges.loc[(((tmp_edges["n1"]==i) & 
                                                                  (tmp_edges["n2"].isin(potential_plex_nodes))) |
                                                                 ((tmp_edges["n2"]==i) & 
                                                                  (tmp_edges["n1"].isin(potential_plex_nodes)))) &
                                                         (tmp_edges["e"]==0)].sort_values(by=['w'])

                cheapest_edge = potential_edges.iloc[:1]
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
    if print_info:
        print(round(time.time()-start, 2), "seconds")
    return (tmp_nodes, tmp_edges, tmp_score) 

    
def SA(solution: Solution, T_init = None, equilibrium = 200, cooling = 0.75, stopping_criteria = "finalT", T_fin_frac = 16, tau=10,
       random_state = 42):
    if random_state != None:
        random.seed(random_state)
    t = 0
    if T_init == None:
        T = solution.get_edges()["w"].abs().sum() # f_max-f_min but f_min = 0
    else:
        T = T_init
    
    if stopping_criteria == "finalT":
        T_fin = T/T_fin_frac
    else:
        T_fin = float('-inf')
        
    if stopping_criteria != "noImprovement":
        tau = float('inf')
        
    global_best_solution = solution
    current_solution = solution
    tmp_changes_since_last_improvement = 0
    start_time = time.time()
    break_flag = False
    
    objective_trajectory = []
    rejected =[]
    temperature =[]
    while (T > T_fin) and (tmp_changes_since_last_improvement <= tau):
        if break_flag:
            break
        print("T:", T)
        print("tmp changes since last improvment:", tmp_changes_since_last_improvement)
        while t<equilibrium:
            nodes, edges, weight = move1node_random(current_solution.get_nodes(),
                                        current_solution.get_edges(),
                                        current_solution.get_weight(),
                                        current_solution.get_s(), add_neg_edges = True,
                                            print_info = False)
            new_solution = Solution(nodes, edges, solution.get_s(), weight)
            if new_solution.get_weight() < current_solution.get_weight():
                current_solution = new_solution
                rejected.append(False)
            else:
                metropolis = math.exp(-abs(new_solution.get_weight()-current_solution.get_weight())/T)
                P = random.uniform(0,1)
                if P < metropolis: # accept solution anyway
                    current_solution = new_solution
                    rejected.append(False)
                else:
                    rejected.append(True)
            t+=1        
            
            # if the (possibly new found) solution is better than the global best
            if current_solution.get_weight() < global_best_solution.get_weight():
                global_best_solution = current_solution
                tmp_changes_since_last_improvement = 0
            
            # get some statistics
            objective_trajectory.append(current_solution.get_weight())
            temperature.append(T)
            
            #check time constraint
            if time.time()-start_time >= 900: # 15 minutes
                break_flag = True
                break
        # cool off
        #print("current score", str(current_solution.get_weight()))
        T = T*cooling
        tmp_changes_since_last_improvement += 1
        t = 0
    return global_best_solution, objective_trajectory, temperature, rejected

def valid(solution: Solution) -> bool:
    plexes = list(solution.get_nodes()["splex"].unique())
    for plex in plexes:
        plex_valid = is_splex(solution.get_nodes(), plex, solution.get_s())
        if not isinstance(plex_valid, (bool)):
            print(plex_valid)
            return False
    return True

def consistent(solution: Solution) -> bool:
    nodes = solution.get_nodes()
    edges = solution.get_edges()
    inconsistent = False

    for index, node in nodes.iterrows():
        impact = edges.loc[((edges["n1"] == node["node_number"])|
                           (edges["n2"] == node["node_number"]))
                          &((edges["e"]==0)&(edges["w"]<=0))| 
                           ((edges["e"]==1)&(edges["w"]>0)), ["w"]].abs().sum().item()

        degree = len(edges.loc[((edges["n1"] == node["node_number"])|
                           (edges["n2"] == node["node_number"]))
                           &(edges["e"]==1)].index)

        #if node["node_impact"].item() != impact:
        #    print(str(node["node_impact"].item()) + " != " + str(impact))
        #    inconsitent = True
        
        if node["current_degree"].item() != degree:
            print(str(node["current_degree"].item()) + " != " + str(degree))
            inconsistent = True

    return not inconsistent