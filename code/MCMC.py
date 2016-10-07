# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:43:36 2016

@author: Ciwan
"""
from graph.node import Node
from graph.graph import Graph
import graph.path_finder as pf
from typing import Dict, List, Iterable, Tuple
import numpy as np
#from problem import Problem
    
class Problem:
    def __init__(self, nrBikers) -> None:
        self.nrNodes = 5
        self.nrBikers = nrBikers
        self.graph = Graph()


class Sim_annSolver:
    rest = 0
    cust = 1
    biker = 2
    
    def __init__(self, data: Problem) -> None:
        """ Read in the problem data and store of use."""
        #Take the order dict and make a List of tuples. The first
        # index is typ type, "biker", "rest" or "cust"
        self.nrNodes        = data.nrNodes
        self.nrBikers       = data.nrBikers
        self.graph          = data.graph
        self.costMatrix     = -1 * np.ones((data.nrNodes, data.nrNodes))
        self.nodeDicts      = List[Dict[int, Node]]
        self.orders         = dict(data.orders)
        self.nrOrders       = 0
        for o in self.orders:
            self.nodeDicts[0][self.nrOrders] = o[0]
            self.nodeDicts[1][self.nrOrders] = o[1]
            self.nrOrders += 1
            
        self.solution       = Dict[int, List[Tuple[int, int]]] 
        self.costOfRoutes   = Dict[int, float]
        self.searchOrder    = List[Tuple[int, int]] #bikers1, biker2
        for i in range(self.nrBikers):
            for j in range(i+1, self.nrBikers):
                self.searchOrder.append((i, j))   
        return None
    
    def runSA() -> None:
        # Do initial heuristic solution, only vor variable number of bikers?
        
        #Initialize cooloing schedule parameters
        
        #lambda interchange + cost of move
        
        #apply acceptance criteria
        
        #update temperatures
        
        #check for stop
        
        return None
        
    def runLambdaDescent() -> None:
        	# Do initial heuristic solution, only vor variable number of bikers?
        
        #lambda interchange + cost of move
        
        #apply acceptance criteria
        
        #check for stop
        return None
    
    def initializeSolution(self) -> None:
        orders = range(self.nrOrders)
        b = 0
        while(len(orders) > 0):
            o = orders.pop()
            self.solution[b].append((rest, o))
            self.solution[b].append((cust, o))
            b += 1
            b = b % self.nrBikers
        for b in range(self.nrBikers):
            self.costOfRoutes[b] = self.calcCostofRoute(self.solution[b])
        return None
            
    def calcCostofRoute(self, route: List[Tuple[int, int]]) -> float:
        cost = 0.0
        for i in range(len(route) - 1):
            node1 = nodeDicts[route[i][0]][route[i][1]]
            node2 = nodeDicts[route[i + 1][0]][route[i + 1][1]]
            cost += self.__getDistance(node1, node2)
        return cost
        
    def lambdaInterchange(self) -> List[Dict[int, List]]:
        """This function implements a 1-interchange mechanism for a carrying 
        capacity of 1 with 1 load per order.
        The function returns the a neighbourhood of solutions to the current 
        solution."""
        
        neighbourhood = []
        for pair in self.searchOrder:
            biker1Route = list(self.solution[pair[0]])
            biker2Route = list(self.solution[pair[1]])
            len1        = len(biker1Route)
            stop1       = len1 + 1 if len1 == 0 else len1
            len2        = len(biker2Route)
            stop2       = len2 + 1 if len2 == 0 else len2
            for inx1 in range(0, stop1, 2):
                for inx2 in range(0, stop2, 2):
                    if(len1 > 0):
                        neighbour = dict(self.solution)
                        lists = self.__01operator(list(biker1Route), 
                                                  list(biker2Route), inx1, inx2)
                        neighbour[pair[0]] = list(lists[0])
                        neighbour[pair[1]] = list(lists[1])
                        neighbourhood.append(dict(neighbour))
                    if(len2 > 0):
                        lists = self.__10operator(list(biker1Route), 
                                                  list(biker2Route), inx1, inx2)
                        neighbour[pair[0]] = list(lists[0])
                        neighbour[pair[1]] = list(lists[1])
                        neighbourhood.append(dict(neighbour))
                    if(len1 > 0 and len2 > 0):
                        lists = self.__11operator(list(biker1Route), 
                                                  list(biker2Route), inx1, inx2)
                        neighbour[pair[0]] = list(lists[0])
                        neighbour[pair[1]] = list(lists[1])
                        neighbourhood.append(dict(neighbour))
                    
        return neighbourhood
    
    """WARNING The following functions determine how orders are transfered between 
    bikers. These needs to be changed to account for capacity != 1 and 
    lambda > 1. """
    def __01operator(self, biker1Route: List, biker2Route: List,
                      inx1: int, inx2: int) -> Tuple[List]:
        r = biker1Route.pop(inx1)
        c = biker1Route.pop(inx1)
        biker2Route.insert(inx2, c)
        biker2Route.insert(inx2, r)
        return biker1Route, biker2Route
    
    def __10operator(self, biker1Route: List, biker2Route: List,
                      inx1: int, inx2: int) -> Tuple[List]:
        r = biker2Route.pop(inx2)
        c = biker2Route.pop(inx2)
        biker1Route.insert(inx1, c)
        biker1Route.insert(inx1, r)
        return biker1Route, biker2Route

    def __11operator(self, biker1Route: List, biker2Route: List,
                      inx1: int, inx2: int) -> Tuple[List]:
        r1 = biker1Route.pop(inx1)
        c1 = biker1Route.pop(inx1)
        r2 = biker2Route.pop(inx2)
        c2 = biker2Route.pop(inx2)        
        biker2Route.insert(inx2, c1)
        biker2Route.insert(inx2, r1)
        biker1Route.insert(inx1, c2)
        biker1Route.insert(inx1, r2)
        return biker1Route, biker2Route
    
    def evaluateCostofMove()
        
    def __getDistance(self, start: Node, goal: Node) -> float:
        # Check matrix
        # Otherwise calc distance and add to matrix
        distance = 0.0
        if(self.costMatrix[start.getID(), goal.getID()] > -1):
            distance = self.costMatrix[start.getID(), goal.getID()]
        else:
            distance = pf.a_star_search(self.graph, start, goal)
            self.costMatrix[start.getID(), goal.getID()] = distance
            self.costMatrix[goal.getID(), start.getID()] = distance
        return distance
        
     
if __name__ == "__main__":
    data = Problem(3)
    simulator = Sim_annSolver(data)
    simulator.solution[0] = [1, 2, 3, 4]
    #simulator.solution[1] = [5, 6]
    #simulator.solution[2] = [7, 8]
    neighbourhood = simulator.lambdaInterchange()
    for o in neighbourhood:
        print(o)