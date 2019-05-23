#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 07:58:28 2019

@author: Zargham
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def getNodeAttributeUniques(g, attr):
    print(type(attr))
    return set(nx.get_node_attributes(g, attr).values())


def make_colors(g):

    node_types = set(nx.get_node_attributes(g, "type").values())

    cmap = plt.cm.jet
    Nc = cmap.N
    Nt = len(node_types)
    dN = int(Nc / Nt)
    cmaplist = [cmap(i * dN) for i in range(Nt)]

    colors = {}
    bunches = {}

    counter = 0
    for nt in node_types:
        bunches[nt] = [x for x, y in g.nodes(data=True) if y["type"] == nt]
        colors[nt] = cmaplist[counter]
        counter = counter + 1

    return colors


def inspectSubGraph(g, bunch, expand=True, verbose=False, label=True, pos="kk"):

    nbunch = bunch
    if expand:
        for s in bunch:
            nbunch = list(nx.all_neighbors(g, s)) + nbunch

    # print(nbunch)
    sg = nx.subgraph(g, set(nbunch))
    colors = make_colors(sg)

    # sgs = nx.get_node_attributes(sg,'shape')
    # print(sgc)
    # nx.draw_kamada_kawai(sg, node_color =sgc, node_shape=sgs , alpha=.5)

    for x, y in sg.nodes(data=True):
        sg.nodes[x]["color"] = colors[y["type"]]
        if y["type"] == "vanilla":
            sg.nodes[x]["label"] = x

        elif (y["type"] == "github/repo") or (y["type"] == "github/user"):
            sg.nodes[x]["label"] = y["address"][4]

        else:
            sg.nodes[x]["label"] = y["type"].split("/")[-1]

    if verbose:
        print("nodes")
        for n in sg.nodes:
            print(n)
            print(sg.nodes[n])

        print("")
        print("edges")
        for e in sg.edges:
            print(e)
            print(sg.edges[e])

    labels = None
    if label:
        labels = nx.get_node_attributes(sg, "label")

    # print(sgc)
    sgc = np.array(list(nx.get_node_attributes(sg, "color").values()))

    if pos == "kk":
        nx.draw_kamada_kawai(
            sg,
            node_color=sgc,
            node_shape=".",
            alpha=0.5,
            labels=labels,
            font_size=8,
            figsize=(20, 10),
        )
    elif pos == "spring":
        nx.draw_spring(
            sg,
            node_color=sgc,
            node_shape=".",
            alpha=0.5,
            labels=labels,
            font_size=8,
            figsize=(20, 10),
        )
    elif pos == "spectral":
        nx.draw_spectral(
            sg,
            node_color=sgc,
            node_shape=".",
            alpha=0.5,
            labels=labels,
            font_size=8,
            figsize=(20, 10),
        )
    else:
        nx.draw_circular(
            sg,
            node_color=sgc,
            node_shape=".",
            alpha=0.5,
            labels=labels,
            font_size=8,
            figsize=(20, 10),
        )
        
    return sg.copy()
