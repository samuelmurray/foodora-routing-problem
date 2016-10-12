from json_parser import *
from graph.path_finder import *
from problem import Problem
import json
from MCMC import SimulatedAnnealing
import numpy as np


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
    with open("data/bikers3.json") as data_file:
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
    solver = SimulatedAnnealing(problem, True)
    """TEST 1: INIT solution """
    solver.initializeSolution()
    print("Is init sol OK?: ", 
          solver.solution == {0: [(0,1),(1,1)], 1: [(0,0), (1,0)], 2: []})
    print("Is init Cost OK?: ", np.sum(np.abs(solver.costOfRoutes - 
                                              np.array([np.sqrt(13), 8.0, 0]))) < 0.05)
    print("-------------\n")
    """Test calcCostOfRoute """
    routes = {0: [(0,1),(1,1)], 1: [(0,0), (1,0)], 2: []}
    print("Routes: ", routes)
    cost1 = solver.calcCostofRoute(routes[0], 0)
    print("Cost1 should be: ", np.sqrt(13), " cost is: ", cost1)
    cost1 = solver.calcCostofRoute(routes[0], 1)
    print("Cost1 should be: ", 2*np.sqrt(13), " cost is: ", cost1)
    cost1 = solver.calcCostofRoute(routes[2], 0)
    print("Cost1 should be: ", 0, " cost is: ", cost1)
    cost1 = solver.calcCostofRoute(routes[1], 1)
    print("Cost1 should be: ", 8, " cost is: ", cost1)
    """neighbourhood print out test """
    neighbourhood = solver.lambdaInterchange()
    print("Neighbourhood is: ", neighbourhood, " OK!")
    # [{(-1, 0), (0, -1), (0, 0)}, {(0, -1)}, {(0, -1)}]
    print("-------------\n")
    """Several calcL tests """
    von = 0
    nach = 2
    routeVon = list(solver.solution[von])
    routeNach = list(solver.solution[nach])
    rest = routeVon.pop(0)
    cust = routeVon.pop(0)   
    l1 = -solver.calcL(routeVon, von, 0, rest, cust)
    print("l1 should be: ", -np.sqrt(13), ", l1 is: ", l1)
    l2 = solver.calcL(routeNach, nach, 0, rest, cust)
    print("l2 should be: ", np.sqrt(13), ", l2 is: ", l2)
    
    von = 0
    nach = 1
    routeVon = list(solver.solution[von])
    routeNach = list(solver.solution[nach])
    rest = routeVon.pop(0)
    cust = routeVon.pop(0)
    l2 = solver.calcL(routeNach, nach, 0, rest, cust)
    print("l2 should be: ", 2*np.sqrt(13), ", l2 is: ", l2)
    
    von = 0
    nach = 1
    routeVon = list(solver.solution[von])
    routeNach = list(solver.solution[nach])
    rest = routeVon.pop(0)
    cust = routeVon.pop(0)   
    l2 = solver.calcL(routeNach, nach, 2, rest, cust)
    print("l2 should be: ", 2*np.sqrt(13) + np.sqrt(10), ", l2 is: ", l2)
    print("-------------\n")
    
    """Test getCostInsDelProcedure """
    print("Solution is : ", solver.solution)
    bikerPair = (0, 1)
    move = (0, 0)
    costChanges, insInx = solver.getCostInsDelProcedure(bikerPair, move)
    print("costChanges should be: ", [8, 2*np.sqrt(13) - 8, 0] , ", is: ", costChanges)
    print("insInx should be: ", -1 , ", is: ", insInx)
    
    bikerPair = (0, 2)
    move = (0, -1)
    costChanges, insInx = solver.getCostInsDelProcedure(bikerPair, move)
    print("costChanges should be: ", [-np.sqrt(13), 0, np.sqrt(13)] , ", is: ", costChanges)
    print("insInx should be: ", 0 , ", is: ", insInx)
    
    bikerPair = (0, 1)
    move = (0, -1)
    costChanges, insInx = solver.getCostInsDelProcedure(bikerPair, move)
    print("costChanges should be: ", [-np.sqrt(13), 2*np.sqrt(13), 0] , ", is: ", costChanges)
    print("insInx should be: ", 0 , ", is: ", insInx)
    
    bikerPair = (0, 1)
    move = (-1, 0)
    costChanges, insInx = solver.getCostInsDelProcedure(bikerPair, move)
    print("costChanges should be: ", [8, -8, 0] , ", is: ", costChanges)
    print("insInx should be: ", 2 , ", is: ", insInx)
    print("-------------\n")
    """Objective function """
    print("Cost should be: ", 8, "cost f is: ", solver.objectiveFunction(solver.costOfRoutes))
    print("newCost should be: ", 8 + np.sqrt(13), "cost f is: ", 
          solver.objectiveFunction(solver.costOfRoutes + costChanges))
    delta = (solver.objectiveFunction(solver.costOfRoutes + costChanges) -
                         solver.objectiveFunction(solver.costOfRoutes))
    print("delta should be: ", np.sqrt(13), "delta is: ", delta)
    
    print("-------------\n")
    solver.initializeCoolingParameters()
    print("Ts should be: ", 2*np.sqrt(13), "Ts is: ", solver.Ts)
    print("Tf should be: ", np.sqrt(13), "Tf is: ", solver.Tf)
    print("alpha should be: ", 5*2, "alpha is: ", solver.alpha)
    print("gamma should be: ", 2, "gamma is: ", solver.gamma)
    print("-------------\n")
    
    oldSol = dict(solver.solution)
    oldCost = np.copy(solver.costOfRoutes)
    print("oldSolution is : ", oldSol)
    print("oldCost is : ", oldCost)
    bikerPair = (0, 1)
    move = (0, 0)
    insInx = -1
    solver.updateSolution(bikerPair, move, insInx)
    print("newSolution should be: ", {0: [(0, 0), (1, 0)], 1: [(0, 1), (1, 1)],
                                      2: []}, "newSol is: ", solver.solution)
    print("cost should be: ", oldCost + np.array([8, 2*np.sqrt(13) - 8, 0])
        , ", is: ", solver.costOfRoutes)
    
    solver.solution = dict(oldSol)
    solver.costOfRoutes = np.copy(oldCost)
    print("oldSolution is : ", oldSol)
    print("oldCost is : ", oldCost)
    bikerPair = (0, 1)
    move = (0, -1)
    insInx = 0
    solver.updateSolution(bikerPair, move, insInx)
    print("newSolution should be: ", {0: [], 1: [(0, 0), (1, 0), (0, 1), (1, 1)],
                                      2: []}, "newSol is: ", solver.solution)
    print("cost should be: ", oldCost + np.array([-np.sqrt(13), 2*np.sqrt(13), 0]) , ", is: ", solver.costOfRoutes)
    #solver.runSA()
    #problem.make_pddl()
