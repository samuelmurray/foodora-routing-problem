from graph.graph import Graph
from graph.node import Node


def graph_from_json(json_data):
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


def orders_from_json(json_data):
    orders = {}
    keys = json_data.keys()  # type: str
    # TODO


def bikers_from_json(json_data):
    bikers = {}
    keys = json_data.keys()  # type: str
    # TODO
