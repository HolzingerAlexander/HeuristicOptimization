from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
import pandas as pd
import time
import random
from helper_functions_assignment2 import*
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets


def train(config: Configuration, instance: str, seed: int = 0) -> float:
	instance_type = "test_instances/"
	problem_instance = instance #"heur002_n_100_m_3274"
	path = "data/"+instance_type+problem_instance+".txt"
	
	print(path)
	
	node_impact_orig, node_degree_orig, plex_assignment, edges_n1, edges_n2, edge_weights, edge_assignment_orig, s, n, m = create_problem_instance(path)
	
	test = GA(pop_size = config["pop_size"],
		  init_no_plexes = config["init_no_plexes"],
		  mutate = config["mutate"],
		  elitism_k = 49,
		  MaxStallGenerations = 2,
		  tolerance = 0.05,
		  node_impact_orig = node_impact_orig,
		  node_degree_orig = node_degree_orig,
		  edge_assignment_orig = edge_assignment_orig, 
		  edge_weights = edge_weights,
		  s = s)
		  
	print("best score:", test.score)

	return test.score

configspace = ConfigurationSpace({"pop_size": (50,150),
				   "init_no_plexes":(3,10),
				   "mutate":[True,False]})

scenario = Scenario(configspace, 
		    output_directory="GA_smac",
		     instances = ["heur002_n_100_m_3274", "heur003_n_120_m_2588"],
		     #instance_features = {"heur002_n_100_m_3274": "heur002", 					   "heur003_n_120_m_2588": "heur003"},
		     deterministic = True, n_trials=10)

smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()

print(incumbent)
