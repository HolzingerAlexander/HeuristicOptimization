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