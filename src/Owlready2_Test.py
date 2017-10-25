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

#%% SPARQL QUERY WITH RDFLIB

from SPARQLWrapper import SPARQLWrapper, JSON

# Query attempt #1
sparql = SPARQLWrapper("http://dbpedia.org/sparql")  # Goes to Virtuoso SPARQL Query Editor site
sparql.setQuery("""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label
    WHERE { <http://dbpedia.org/resource/Asturias> rdfs:label ?label }
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    print(result["label"]["value"])

# Query attempt #2
sparql.setQuery("""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>   
                SELECT *
                  WHERE
                    {
                      ?e <http://dbpedia.org/ontology/series>         <http://dbpedia.org/resource/The_Sopranos>  .
                      ?e <http://dbpedia.org/ontology/releaseDate>    ?date                                       .
                      ?e <http://dbpedia.org/ontology/episodeNumber>  ?number                                     .
                      ?e <http://dbpedia.org/ontology/seasonNumber>   ?season
                    }
                  ORDER BY DESC(?date)
                  """)

sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    print('Name: ' + result['e']['value'])
    print('Season ' + result['season']['value'] + ', Episode ' + result['number']['value'])
    print('Release date: ' + result['date']['value'])
    print()
