#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:41:24 2017

@author: adrienne
"""
#%% SET-UP

import os
entry_point = r'/home/adrienne/Code/PythonProjects/CFPB/src'
os.chdir(entry_point)

#%% GET ONTOLOGY

from owlready2 import *
onto_path.append(r'/home/adrienne/Code/PythonProjects/CFPB/ontologies')
onto = get_ontology("http://www.lesfleursdunormal.fr/static/_downloads/pizza_onto.owl")
onto.load()

#%% EXPERIMENT

list(onto.classes())
list(onto.individuals())  
list(onto.object_properties()) 
list(onto.data_properties())
list(onto.annotation_properties())
list(onto.disjoint_classes())

# searching for all entities with an IRI ending with ‘Topping’:
onto.search(iri = "*Topping")

# searches for all entities that are related to another one with the ‘has_topping’ relation:
onto.search(has_topping = "*")
