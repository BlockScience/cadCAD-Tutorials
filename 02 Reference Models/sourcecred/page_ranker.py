#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:43:21 2019

@author: Zargham
"""

import networkx as nx
import pandas as pd
import numpy as np

#defaults
default_self_loop_wt= .001 

def update_score(g,alpha,seed, lazy=False, lazy_wt = .5):
    
    #lazy random walk assumes a topology independent 1/2 wt on self-loops
    lazy_wt = lazy_wt*float(lazy) 
    
    prior_x = nx.get_node_attributes(g,'score')
    for n in g.nodes:
        self_wt = g.nodes[n]['self_wt']/g.nodes[n]['total_wt']
        
        val = (1-alpha)*self_wt*prior_x[n] + alpha*seed[n]
        for nb in g.nodes[n]['out_nbr']:
            #outbound neighbor
            e_count = edge_count(g, n,nb)
            for e3 in range(e_count):
                wt = g.edges[(n,nb,e3)]['out_weight']/g.nodes[nb]['total_wt']
                val = val + (1-alpha)*wt*prior_x[nb]
        
        for nb in g.nodes[n]['in_nbr']:
            #inbound neighbor
            e_count = edge_count(g, nb,n)
            for e3 in range(e_count):
                wt = g.edges[(nb,n,e3)]['in_weight']/g.nodes[nb]['total_wt']
                val = val + (1-alpha)*wt*prior_x[nb]
                    
        #print(val)
                    
        g.nodes[n]['score']= lazy_wt*prior_x[n]+(1-lazy_wt)*val
    
    return g

#helper function
def edge_count(g,src,dst):
    i =0
    stop = False
    while not(stop):
        try:
            g.edges[(src,dst,i)]
            i=i+1
        except:
            stop = True
            return i

#tuples are (to_weight, from_weight)
default_edge_wt_by_type = {
    'github/authors': (0.5,1),
    'github/hasParent':(1,1/4),
    'git/hasParent':(1,1/4),
    'github/mentionsAuthor': (1,1/32),
    'github/mergedAs':(.5,1),
    'github/references':(1,1/16),
    'github/reactsHeart':(2,1/32),
    'github/reactsHooray':(4,1/32),
    'github/reactsRocket':(1,0), #appears to be missing from current implementation
    'github/reactsThumbsUp':(1,1/32)
    }

default_node_wt_by_type = {
    'github/issue':2.0, 
    'github/repo':4.0, 
    'github/comment': 1.0, 
    'git/commit':2.0, 
    'github/user':1.0,
    'github/bot':1.0,
    'github/review': 1.0, 
    'github/pull': 4.0
    }


def wt_heuristic(g,
                 node_wt_by_type=default_node_wt_by_type,
                 edge_wt_by_type=default_edge_wt_by_type,
                 self_loop_wt=default_self_loop_wt):
    
    for e in g.edges:
        e_wts = edge_wt_by_type[g.edges[e]['type']]
        src_wt = node_wt_by_type[g.nodes[e[0]]['type']]
        dst_wt = node_wt_by_type[g.nodes[e[1]]['type']]
        
        g.edges[e]['in_weight'] = e_wts[0]*dst_wt
        g.edges[e]['out_weight'] = e_wts[1]*src_wt
    
    '''
    for n in g.nodes:
        wt = self_loop_wt
        for nb in nx.all_neighbors(g,n):
            #outbound neighbor
            if nb in g.neighbors(n):
                e_count = edge_count(g,n,nb)
                for e3 in range(e_count):
                    wt = wt + g.edges[(n,nb,e3)]['out_weight']
            #inbound neighbor
            else:
                e_count = edge_count(g,nb,n)
                for e3 in range(e_count):
                    wt = wt + g.edges[(nb,n,e3)]['in_weight']
                
        g.nodes[n]['denominator']=wt
    '''
    
    #create neighborhoods
    for n in g.nodes:
        g.nodes[n]['all_nbr']= set(nx.all_neighbors(g,n))
        g.nodes[n]['in_nbr'] = set()
        g.nodes[n]['out_nbr'] = set()
        for nb in g.nodes[n]['all_nbr']:
            #print((n,nb))
            try :
                g.edges[(nb,n,0)]
                g.nodes[n]['in_nbr'].add(nb)
            except:
                pass
            try :
                g.edges[(n,nb,0)]
                g.nodes[n]['out_nbr'].add(nb)
            except:
                pass
    
    for n in g.nodes:
        self_wt = self_loop_wt#/g.nodes[n]['denominator']
        g.nodes[n]['self_wt']=self_wt
        total_wt = self_wt
        for nb in g.nodes[n]['out_nbr']:
            #outbound neighbor
            e_count = edge_count(g, n,nb)
            for e3 in range(e_count):
                wt = g.edges[(n,nb,e3)]['in_weight']#/g.nodes[nb]['denominator']
                #g.edges[(n,nb,e3)]['normalized_out_wt']=wt
                total_wt = total_wt+wt
            
        for nb in g.nodes[n]['in_nbr']:
           #inbound neighbor
            e_count = edge_count(g, nb,n)
            for e3 in range(e_count):
                wt = g.edges[(nb,n,e3)]['out_weight']#/g.nodes[nb]['denominator']
                #g.edges[(nb,n,e3)]['normalized_in_wt']=wt
                total_wt = total_wt+wt
        
        
        g.nodes[n]['total_wt'] = total_wt
        
    return g

def pageRanker(g,
               alpha,
               K,
               seed=None,
               initial_value = None,
               lazy=False,
               lazy_wt = .5,
               lazy_decay = True,
               self_loop_wt=default_self_loop_wt,
               node_wt_by_type =default_node_wt_by_type,
               edge_wt_by_type=default_edge_wt_by_type):
    
    #improve input verification for seed
    #must be dict keyed to nodes
    #with non-negative floating point values summing to 1
    if seed==None:
        N = len(g.nodes)
        seed = {n:1.0/N for n in g.nodes}
    
    #improve input verification for initial value
    #must be dict keyed to nodes
    #with non-negative floating point values summing to 1
    if initial_value==None:
        initial_value = seed

    for n in g.nodes:    
        g.nodes[n]['score'] = initial_value[n]    
    
    g = wt_heuristic(g,
                     node_wt_by_type=node_wt_by_type,
                     edge_wt_by_type=edge_wt_by_type,
                     self_loop_wt=self_loop_wt)
    
    #print(g.nodes[0])
    
    x_dict = {0:initial_value}
    for k in range(0,K):
        g = update_score(g,
                         alpha,
                         seed,
                         lazy,
                         lazy_wt*(1-int(lazy_decay)*k/(k+3)))
        x_dict[k+1] = nx.get_node_attributes(g,'score')
    
    
    #result in numpy array format
    pr= np.array([g.nodes[n]['score'] for n in range(len(g.nodes))])
    
    #trajectory in pandas dataframe format
    df = pd.DataFrame(x_dict).T
    return pr,df, g