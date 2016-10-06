from graph.graph import Graph, graph_from_json
from graph.node import Node
from graph.path_finder import *
import json


def example_graph():
    graph = Graph()
    oster = Node(1, 2)
    soder = Node(3, 2)
    norr = Node(5, 4)
    graph.add_node(oster)
    graph.add_node(soder)
    graph.add_node(norr)
    graph.add_edge(oster, soder)
    graph.add_edge(norr, oster)
    edges = graph.edges()

    print(graph)
    print()
    for neighbour in graph.neighbours(0):
        print(neighbour)
    for neighbour in graph.neighbours(1):
        print(neighbour)
    for neighbour in graph.neighbours(2):
        print(neighbour)


if __name__ == '__main__':
    with open("graph.json") as data_file:
        data = json.load(data_file)
        graph = graph_from_json(data)
    print(graph)
    start = Node.node_by_name("soder")
    goal = Node.node_by_name("vasastan")
    path, cost = a_star_search(graph, start, goal)
    for node in path:
        print(node)
    print()
    print("Min cost to go from soder to vasastan is {cost}".format(cost=cost))
