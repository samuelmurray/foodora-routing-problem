""" Functions to parse the json files and generate correct data structures """

from graph.graph import Graph
from graph.node import Node
from typing import Dict, Tuple


def graph_from_json(json_data) -> Graph:
    graph = Graph()
    keys = json_data.keys()  # type: str
    for key in keys:
        x = json_data[key]["x"]
        y = json_data[key]["y"]
        graph.add_node(Node(x, y, key))
    for key in keys:
        neighbours = json_data[key]["neighbours"]
        for neighbour in neighbours:
            graph.add_edge(Node.node_by_name(key), Node.node_by_name(neighbour))
    return graph


def orders_from_json(json_data) -> Dict[int, Tuple[Node, Node]]:
    orders = {}
    keys = json_data.keys()  # type: str
    for i, key in enumerate(keys):
        from_node = Node.node_by_name(json_data[key]["from"])
        to_node = Node.node_by_name(json_data[key]["to"])
        orders[i] = (from_node, to_node)
    return orders


def bikers_from_json(json_data) -> Dict[int, Node]:
    bikers = {}
    keys = json_data.keys()  # type: str
    for i, key in enumerate(keys):
        bikers[i] = Node.node_by_name(json_data[key])
    return bikers
