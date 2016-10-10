"""
A graph class. The graph consists of nodes and a dictionary that represent edges.
"""

from typing import Dict, List, Iterable, Tuple
from graph.node import Node


class Graph:
    def __init__(self, graph_dict: Dict[Node, List[Node]] = None) -> None:
        """ initializes a graph object
            If no dictionary or None is given,
            an empty dictionary will be used
        """
        if graph_dict is None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def nodes(self) -> Iterable(Node):
        """ returns the nodes in the graph """
        return list(self.__graph_dict.keys())

    def edges(self) -> Iterable[Tuple[Node]]:
        """ returns the edges in the graph """
        return self.__generate_edges()

    def neighbours(self, node: Node) -> Iterable[Node]:
        """ return the neighbours of a node """
        if node is None:
            return None
        for neighbour in self.__graph_dict[node]:
            yield neighbour

    def add_node(self, node: Node):
        """ If the node "node" is not in
            self.__graph_dict, a key "node" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if node not in self.__graph_dict:
            self.__graph_dict[node] = []

    def add_edge(self, node1: Node, node2: Node):
        """ Adds an edge between two nodes. Since edges are not directed, it is added in both directions. """
        self.__add_edge(node1, node2)
        self.__add_edge(node2, node1)

    def __add_edge(self, node1: Node, node2: Node):
        if node2 in self.__graph_dict[node1]:  # Don't add duplicate edges
            return
        if node1 in self.__graph_dict:
            self.__graph_dict[node1].append(node2)
        else:
            self.__graph_dict[node1] = [node2]

    def __generate_edges(self) -> Iterable[Tuple[Node, Node]]:
        """ A method generating the edges of the graph. Edges are represented as tuples of two nodes. """
        edges = []
        for node in self.__graph_dict.keys():
            for neighbour in self.__graph_dict[node]:
                # Since all edges go both ways, we need only return one of them.
                if {neighbour, node} not in edges:
                    edges.append({node, neighbour})
                    yield (node, neighbour)

    def __str__(self) -> str:
        res = "nodes: "
        for k in self.__graph_dict.keys():
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += "{" + edge[0].name() + "-" + edge[1].name() + "} "
        return res
