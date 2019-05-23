from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import collections

AddressType = collections.namedtuple("AddressType", ("prefix", "type"))


def _type_prefix_match(address_types, address):
    """For a given address, find the type matching the address.

    Takes an object containing an array of {prefix, type} pairs, and
    an address. Returns the first type whose corresponding prefix
    was a prefix of the given address.
    """
    for address_type in address_types:
        prefix = address_type.prefix
        if address[: len(prefix)] == prefix:
            return address_type.type
    raise ValueError("No matching prefix for {}".format(address))


def node_type(address):
    """Return a string that identifies the "type" of a SourceCred node.

    For any anticipated SourceCred node (i.e., it was supplied by one of the two
    standard SourceCred plugins, i.e., sourcecred/git and sourcecred/github),
    this method returns a string which identifies the most specific declared
    node_type that matches the given node.

    The SourceCred type system is still pretty ad-hoc,
    (see: https://github.com/sourcecred/sourcecred/issues/710), so this system is
    likely to change in the future.
    """
    NODE_PREFIX_TO_TYPE = [
        AddressType(prefix=["sourcecred", "github", "REPO"], type="github/repo"),
        AddressType(
            prefix=["sourcecred", "github", "USERLIKE", "USER"], type="github/user"
        ),
        AddressType(
            prefix=["sourcecred", "github", "USERLIKE", "BOT"], type="github/bot"
        ),
        AddressType(prefix=["sourcecred", "github", "PULL"], type="github/pull"),
        AddressType(prefix=["sourcecred", "github", "ISSUE"], type="github/issue"),
        AddressType(prefix=["sourcecred", "github", "REVIEW"], type="github/review"),
        AddressType(prefix=["sourcecred", "github", "COMMENT"], type="github/comment"),
        AddressType(prefix=["sourcecred", "git", "COMMIT"], type="git/commit"),
    ]
    return _type_prefix_match(NODE_PREFIX_TO_TYPE, address)


def edge_type(address):
    """Return a string that identifies the "type" of a SourceCred edge.

    For any anticipated SourceCred edge (i.e., it was supplied by one of the two
    standard SourceCred plugins, i.e., sourcecred/git and sourcecred/github),
    this method returns a string which identifies the most specific declared
    edge_type that matches the given node.

    The SourceCred type system is still pretty ad-hoc,
    (see: https://github.com/sourcecred/sourcecred/issues/710), so this system is
    likely to change in the future.
    """
    EDGE_PREFIX_TO_TYPE = [
        AddressType(
            prefix=["sourcecred", "github", "HAS_PARENT"], type="github/hasParent"
        ),
        AddressType(
            prefix=["sourcecred", "github", "REFERENCES"], type="github/references"
        ),
        AddressType(
            prefix=["sourcecred", "github", "MENTIONS_AUTHOR"],
            type="github/mentionsAuthor",
        ),
        AddressType(prefix=["sourcecred", "github", "AUTHORS"], type="github/authors"),
        AddressType(prefix=["sourcecred", "github", "PULL"], type="github/pull"),
        AddressType(prefix=["sourcecred", "github", "ISSUE"], type="github/issue"),
        AddressType(prefix=["sourcecred", "github", "REVIEW"], type="github/review"),
        AddressType(prefix=["sourcecred", "github", "COMMENT"], type="github/comment"),
        AddressType(
            prefix=["sourcecred", "github", "MERGED_AS"], type="github/mergedAs"
        ),
        AddressType(
            prefix=["sourcecred", "github", "REACTS", "HOORAY"],
            type="github/reactsHooray",
        ),
        AddressType(
            prefix=["sourcecred", "github", "REACTS", "THUMBS_UP"],
            type="github/reactsThumbsUp",
        ),
        AddressType(
            prefix=["sourcecred", "github", "REACTS", "HEART"],
            type="github/reactsHeart",
        ),
        AddressType(
            prefix=["sourcecred", "github", "REACTS", "ROCKET"],
            type="github/reactsRocket",
        ),
        AddressType(prefix=["sourcecred", "git", "HAS_PARENT"], type="git/hasParent"),
    ]
    return _type_prefix_match(EDGE_PREFIX_TO_TYPE, address)


def json_to_graph(json):
    """Convert a serialized SourceCred graph to a MultiDiGraph.

    Takes in a Python dict representing a SourceCred graph json.
    Returns a networkx MultiDiGraph, with node and edge type identifiers
    added as an additional property.
    """
    [compat, data] = json
    assert compat["type"] == "sourcecred/graph", compat
    assert compat["version"] == "0.4.0", compat

    def nodePropertyDict(address):
        return {"address": tuple(address), "type": node_type(address)}

    def edgePropertyDict(address):
        return {"address": tuple(address), "type": edge_type(address)}

    nodes = data["nodes"]
    edges = data["edges"]
    g = nx.MultiDiGraph()
    for (i, n) in enumerate(nodes):
        g.add_node(i, **nodePropertyDict(n))
    for e in edges:
        g.add_edge(e["srcIndex"], e["dstIndex"], **edgePropertyDict(e["address"]))
    return g
