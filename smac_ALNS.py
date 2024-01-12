from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
import pandas as pd
import time
import random
from helper_functions_assignment2 import*
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
import os
import re

#get all instances
folder_path = 'data/inst_tuning/'
problem_instances = [file.replace('.txt', '') for file in os.listdir(folder_path) if file.endswith('.txt')]

# get the instance features, i.e. get n and m
pattern = r"heur\d{3}_n_(\d+)_m_(\d+)"
inst_features = {}
for inst in problem_instances:
	match = re.match(pattern, inst)
	if match:
		n_number = int(match.group(1))
		m_number = int(match.group(2))
		inst_features[inst] = [n_number, m_number]

# define the training function that takes the parameters that are being tuned and the instance
# apparently it also needs a seed
def train(config: Configuration, instance: str, seed: int = 0) -> float:
	instance_type = 'data/inst_tuning/'
	problem_instance = instance #"heur002_n_100_m_3274"
	path = instance_type+problem_instance+".txt"
	
	print(path)
	
	node_impact_orig, node_degree_orig, plex_assignment, edges_n1, edges_n2, edge_weights, edge_assignment_orig, s, n, m = create_problem_instance(path)
	plex_assignment = np.random.choice(10, n)
	number_of_phases = round(4000/config["iterations_per_phase"])
	repair_solution(node_impact_orig, node_degree_orig, plex_assignment, edge_weights, edge_assignment_orig, s)
    
	start = time.time()
	
	_, _, _, _, score = ALNS(min_weight = config["min_weight"],
                             reaction_factor = config["reaction_factor"],
                             iterations_per_phase = config["iterations_per_phase"],
                             number_of_phases = number_of_phases, 
                             node_impact = node_impact_orig,
                             node_degree = node_degree_orig,
                             edge_assignment = edge_assignment_orig,
                             edge_weights = edge_weights, 
                             plex_assignment = plex_assignment, 
                             s = s,
							 trajectory= False)

	
	runtime = time.time()-start 
	print("best score:", score, "runtime:", runtime)

	return {'score': score, 'runtime': runtime}
	
if __name__ == "__main__":

	configspace = ConfigurationSpace({"min_weight": (0.01,0.25),
					   "reaction_factor":(0.01,1.0),
					   "iterations_per_phase":(5,200)})

	scenario = Scenario(configspace, 
			     output_directory="ALNS_smac",
			     instances = problem_instances, 
			     instance_features = inst_features,
			     objectives=["score", "runtime"],
			     walltime_limit=28800,
			     deterministic = True, 
			     n_trials=100, # how many runs in total
			     min_budget=1,  # Use min one instance
			     max_budget=18#, # use max 10 instances
			     #n_workers = 4 # use 4 workers in parallel
			     )

	smac = HyperparameterOptimizationFacade(scenario, train)
	incumbent = smac.optimize()

	print(incumbent)
