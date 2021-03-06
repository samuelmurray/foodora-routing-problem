""" Main file to run the different algorithms """

from json_parser import *
from graph.path_finder import *
from problem import Problem
import json
from MCMC import SimulatedAnnealing


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


def print_solution(graph: Graph):
    print("Found solution with cost = {0:0.1f}".format(solver.bestCost))
    for i in solver.bikerStart:
        biker_start = solver.bikerStart[i]
        nr_orders = len(solver.bestSolution[i])//2
        if nr_orders == 0:
            print("Biker", i, "does not take any orders. Is at", biker_start.name())
        for j in range (0, nr_orders):
            order_id = solver.bestSolution[i][j+1][1]
            start = solver.nodeDicts[0][order_id]
            goal = solver.nodeDicts[1][order_id]
            path_init, cost_init = a_star_search(graph, biker_start, start)
            path, cost = a_star_search(graph, start, goal)
            #print(solver.bestSolution[i])
            print("Biker", i, "starts at", biker_start.name(),
                  ", picks up order", order_id,
                  "at", start.name(), "by taking route with cost = {0:0.1f}".format(cost_init)) 
            for node_init in path_init:
                print(node_init.name())
            print("and goes to customer at", 
                    goal.name(), ". Min cost to go from", start.name(), 
                    "to", goal.name(), 
            "is {0:0.1f}".format(cost), ". \nThe path is:")
            for node in path:
                print(node.name())
            biker_start = goal
    

def runJustSA(problem):
    solver = SimulatedAnnealing(problem, True)
    solver.runSA()
    return solver


if __name__ == '__main__':
    
    problem, solver = run()
    print_solution(problem.graph())
    problem.make_pddl()
