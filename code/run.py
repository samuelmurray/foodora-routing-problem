from graph.graph import Graph
from json_parser import *
from graph.node import Node
from graph.path_finder import *
from problem import Problem
import json
from MCMC import SimulatedAnnealing


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


def run():
    with open("data/graph.json") as data_file:
        data = json.load(data_file)
        graph = graph_from_json(data)
    with open("data/orders15.json") as data_file:
        data = json.load(data_file)
        orders = orders_from_json(data)
    with open("data/bikers10.json") as data_file:
        data = json.load(data_file)
        bikers = bikers_from_json(data)
    print("GRAPH")
    for node in graph.nodes():
        print("\nNode {} has the following neighbours:".format(node.name()))
        for neighbour in graph.neighbours(node):
            print(neighbour.name())

    print("BIKERS")
    for key in bikers.keys():
        print("Id: {id}. Location: {node}".format(id=key, node=bikers[key].name()))
    print("ORDERS")
    for key in orders.keys():
        print("Id: {id}. From: {from_node}, To: {to_node}".format(id=key,
                                                                  from_node=orders[key][0].name(),
                                                                  to_node=orders[key][1].name()))
    print("-------------\n")
    problem = Problem(graph, bikers, orders)

    solver = SimulatedAnnealing(problem, True)
    solver.runSA()
    #print("Found solution: ", solver.bestSolution)
    #print("Solution cost: ", solver.bestCost)
    #print("Cost for all bikers: ", solver.bestCostOfRoutes)

    return problem, solver
    
def print_solution(graph:Graph):
    print("Found solution with cost = ", solver.bestCost)
    for index in solver.bikerStart:
        print("Biker ", index, " starts at ", solver.bikerStart[index].name())        
    for i in solver.bikerStart:
        nr_orders = len(solver.bestSolution[i])//2
        for j in range (0, nr_orders):
            order_id = solver.bestSolution[i][j+1][1]
            start = solver.nodeDicts[0][order_id]
            goal = solver.nodeDicts[1][order_id]
            path, cost = a_star_search(graph, start, goal)
            #print(solver.bestSolution[i])
            print("Biker ", i, "picks up order ", order_id,
                  "at ", start.name(), 
                    "and goes to customer at", 
                    goal.name())
            print("Min cost to go from", start.name(), "to", goal.name(), 
            "is", cost, ". The path is:")
            for node in path:
                print(node.name())
    

if __name__ == '__main__':
    
    problem, solver = run()
    print_solution(problem.graph())
    problem.make_pddl()

