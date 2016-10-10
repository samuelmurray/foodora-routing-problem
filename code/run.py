from graph.graph import Graph
from json_parser import *
from graph.node import Node
from graph.path_finder import *
from problem import Problem
import json
from MCMC import SimulatedAnnealing


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


def test_astar():
    with open("data/graph.json") as data_file:
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


if __name__ == '__main__':
    with open("data/graph.json") as data_file:
        data = json.load(data_file)
        graph = graph_from_json(data)
    with open("data/orders.json") as data_file:
        data = json.load(data_file)
        orders = orders_from_json(data)
    with open("data/bikers.json") as data_file:
        data = json.load(data_file)
        bikers = bikers_from_json(data)
    print("BIKERS")
    for key in bikers.keys():
        print("Id: {id}. Location: {node}".format(id=key, node=bikers[key].name()))
    print("ORDERS")
    for key in orders.keys():
        print("Id: {id}. From: {from_node}, To: {to_node}".format(id=key,
                                                                  from_node=orders[key][0].name(),
                                                                  to_node=orders[key][1].name()))

    problem = Problem(graph, bikers, orders)
    solver = SimulatedAnnealing(problem)
    #problem.make_pddl()
