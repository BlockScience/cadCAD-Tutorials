#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:00:33 2019

@author: Zargham
"""
import networkx as nx

def lineGraphGen(N, bidir=False,nodeTypeName='vanilla',edgeTypeName='vanilla'):

    line = nx.path_graph(N, create_using=nx.MultiDiGraph)
    if not(bidir):
        G = line
    else:  
        edges = line.edges
        G = nx.MultiDiGraph()
        for e in edges:
            G.add_edge(e[0],e[1])
            G.add_edge(e[1],e[0])
    
    nx.set_node_attributes(G,nodeTypeName, 'type')
    nx.set_edge_attributes(G,edgeTypeName, 'type')
    
    return G

def starGraphGen(N, kind='sink',nodeTypeName='vanilla',edgeTypeName='vanilla'):
    
    star = nx.star_graph(N)
    G = nx.MultiDiGraph()

    for e in star.edges:
        if (kind == 'source') or (kind == 'bidir'):
            G.add_edge(e[0],e[1])
        if (kind == 'sink') or (kind == 'bidir'):
            G.add_edge(e[1],e[0])

    nx.set_node_attributes(G,nodeTypeName, 'type')
    nx.set_edge_attributes(G,edgeTypeName, 'type')
    
    return G

def circleGraphGen(N, bidir=False,nodeTypeName='vanilla',edgeTypeName='vanilla' ):
    
    circle = nx.cycle_graph(N, create_using=nx.MultiDiGraph)
    if not(bidir):
        G = circle
    else:  
        edges = circle.edges
        G = nx.MultiDiGraph()
        for e in edges:
            G.add_edge(e[0],e[1])
            G.add_edge(e[1],e[0])
            
    nx.set_node_attributes(G,nodeTypeName, 'type')
    nx.set_edge_attributes(G,edgeTypeName, 'type')
    
    return G

def treeGraphGen(r,h, kind='sink',nodeTypeName='vanilla',edgeTypeName='vanilla'):
    
    tree = nx.balanced_tree(r,h, create_using=nx.MultiDiGraph)
    
    if kind=='source':
        G = tree
    elif kind =='sink':
        G = nx.MultiDiGraph()
        for e in tree.edges:
            G.add_edge(e[1],e[0])
    elif kind == 'bidir':
        G = nx.MultiDiGraph()
        for e in tree.edges:
            G.add_edge(e[1],e[0])
            G.add_edge(e[0],e[1])

    nx.set_node_attributes(G,nodeTypeName, 'type')
    nx.set_edge_attributes(G,edgeTypeName, 'type')
    
    return G    