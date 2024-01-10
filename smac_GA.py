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
	
	start = time.time()
	test = GA(pop_size = config["pop_size"],
		  init_no_plexes = config["init_no_plexes"],
		  mutate = config["mutate"],
		  elitism_k = config["elitism_k"],
		  MaxStallGenerations = config["MaxStallGenerations"],
		  tolerance = config["tolerance"],
		  node_impact_orig = node_impact_orig,
		  node_degree_orig = node_degree_orig,
		  edge_assignment_orig = edge_assignment_orig, 
		  edge_weights = edge_weights,
		  s = s)
	
	runtime = time.time()-start 
	print("best score:", test.score, "runtime:", runtime)

	return {'score': test.score, 'runtime': runtime}
	
if __name__ == "__main__":

	configspace = ConfigurationSpace({"pop_size": (50,150),
					   "init_no_plexes":(3,25),
					   "mutate":[True,False],
					   "elitism_k":(0,50),
					   "MaxStallGenerations": (0,5),
					   "tolerance": (0.01,0.07)})

	scenario = Scenario(configspace, 
			     output_directory="GA_smac",
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
