# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:43:36 2016

@author: Ciwan
"""
from graph.node import Node
from graph.graph import Graph
import graph.path_finder as pf
from typing import Dict, List, Tuple, Set
import numpy as np
from warnings import warn
#from problem import Problem


class Problem:
    def __init__(self, nrBikers) -> None:
        self.nrNodes = 5
        self.nrBikers = nrBikers
        self.graph = Graph()
        self.orders = dict()
        for i in range(10):
            self.orders[i] = (Node(0.5 * i, 0.5 * i), Node(-0.5 * i, -0.5 * i))


class Sim_annSolver:
    REST = 0
    CUST = 1
    
    def __init__(self, data: Problem) -> None:
        """ Read in the problem data and store of use."""
        # Take the order dict and make a List of tuples. The first
        # index is typ type, "biker", "REST" or "CUST"
        self.nrNodes        = data.nrNodes
        self.nrBikers       = data.nrBikers
        self.graph          = data.graph
        self.costMatrix     = -1 * np.ones((data.nrNodes, data.nrNodes))
        self.nodeDicts      = [dict(), dict()]  # type: List[Dict[int, Node]]
        self.orders         = dict(data.orders)
        self.nrOrders       = 0
        for o in range(len(self.orders)):
            self.nodeDicts[0][self.nrOrders] = self.orders[o][0]
            self.nodeDicts[1][self.nrOrders] = self.orders[o][1]
            self.nrOrders += 1

        self.solution       = dict()  # type: Dict[int, List[Tuple[int, int]]]
        self.costOfRoutes   = np.inf * np.ones(self.nrBikers)
        # bikers1, biker2
        self.searchOrder    = []  # type: List[Tuple[int, int]]
        for i in range(self.nrBikers):
            self.solution[i] = []
            for j in range(i + 1, self.nrBikers):
                self.searchOrder.append((i, j))
        self.Ts = None  # type: float
        self.Tf = None  # type: float
        self.Tr = None  # type: float
        self.alpha = None  # type: int
        self.gamma = None  # type: int
        self.bestSolution = None  # type: Dict[int, List[Tuple[int, int]]]

    def runSA(self, R=3) -> None:
        # Do initial heuristic solution, only vor variable number of bikers?
        self.initializeSolution()
        # Initialize cooloing schedule parameters
        nrResets = 0
        self.bestSolution = dict(self.solution)
        self.bestCost = self.objectiveFunction(self.costOfRoutes)
        self.Tb = -1
        self.initializeCoolingParameters()
        k = 1
        Tk = self.Ts
        while nrResets < R:
            foundNewSol = False
            self.searchOrder = np.random.permutation(self.searchOrder)
            # lambda interchange + cost of move
            neighbourhood = self.lambdaInterchange()  # type: List[Set[Tuple[int, int]]]
            for n in range(len(self.searchOrder)):
                bikerPair = self.searchOrder[n]
                moveSet = set(neighbourhood[n])
                nrMoves = len(moveSet)
                for m in range(nrMoves):
                    move = moveSet.pop()
                    costChanges, insInx = self.getCostInsDelProcedure(bikerPair, move)
                    delta = (self.objectiveFunction(self.costOfRoutes + costChanges) -
                             self.objectiveFunction(self.costOfRoutes))
                    theta = np.random.random()
                    if (delta <= 0) or (np.exp(-delta / Tk) > theta):
                        self.updateSolution(bikerPair, move)
                        newCost = self.objectiveFunction(self.costOfRoutes)
                        if newCost < self.bestCost:
                            self.bestSolution = self.solution
                            self.bestCost = newCost
                            self.Tb = Tk
                            nrResets = 0
                        else:
                            pass
                        beta = ((self.Ts - self.Tf) / ((self.alpha + self.gamma * np.sqrt(k)) * self.Ts * self.Tf))
                        Tk = Tk / (1 + beta * Tk)
                        foundNewSol = True
                        break
                    else:
                        pass
                if foundNewSol:
                    break
                else:
                    pass
            else:
                self.Tr = np.max(self.Tr / 2, self.Tb)
                Tk = self.Tr
                nrResets += 1
            k += 1

        # apply acceptance criteria
        # update temperatures
        # check for stop

    def initializeCoolingParameters(self) -> None:
        # lambda interchange + cost of move
        deltaMax = 0
        deltaMin = np.inf
        Nfeas = 0
        neighbourhood = self.lambdaInterchange()  # type: List[Set[Tuple[int, int]]]
        # Now we have a 1-interchange neighbourhood, a set of possible moves
        for n in range(len(self.searchOrder)):
            bikerPair = self.searchOrder[n]
            moveSet = set(neighbourhood[n])
            nrMoves = len(moveSet)
            for m in range(nrMoves):
                move = moveSet.pop()
                costChanges, insInx = self.getCostInsDelProcedure(bikerPair, move)
                if self.checkFeasibility(bikerPair, move):
                    Nfeas += 1
                delta = (self.objectiveFunction(self.costOfRoutes + costChanges) -
                         self.objectiveFunction(self.costOfRoutes))
                if np.abs(delta) < deltaMin:
                    deltaMin = np.abs(delta)
                if np.abs(delta) > deltaMax:
                    deltaMax = np.abs(delta)
        self.Ts = deltaMax
        self.Tf = deltaMin
        self.Tr = self.Ts
        self.alpha = self.nrOrders * Nfeas
        self.gamma = self.nrOrders

    def runLambdaDescent(self, thres=0.0) -> Dict[int, List[Tuple[int, int]]]:
        # Do initial heuristic solution, only vor variable number of bikers?
        self.initializeSolution()
        # Now we have an initial self.solution
        # and en initial self.costOfRoutes
        done = False
        while not done:
            done = True
            # lambda interchange + cost of move
            neighbourhood = self.lambdaInterchange()  # type: List[Set[Tuple[int, int]]]
            # Now we have a 1-interchange neighbourhood, a set of possible moves
            for n in range(len(self.searchOrder)):
                bikerPair = self.searchOrder[n]
                moveSet = set(neighbourhood[n])
                nrMoves = len(moveSet)
                for m in range(nrMoves):
                    move = moveSet.pop()
                    costChanges, insInx = self.getCostInsDelProcedure(bikerPair, move)
                    delta = (self.objectiveFunction(self.costOfRoutes + costChanges) -
                             self.objectiveFunction(self.costOfRoutes))
                    if delta < thres:
                        self.updateSolution(bikerPair, move, insInx)
                        done = False
                        break  # Break 2nd for loop
                if not done:
                    break  # Break first for loop, but coninute while loop
                else:
                    continue  # No better found, continute 1st for loop

        return dict(self.solution)

    def initializeSolution(self) -> None:
        """Creates and initial feasible solution. Takes all the orders and 
        distributes them roughly equal among the bikers. Every biker goes 
        directly from the restaurant to the customer."""
        orders = list(range(self.nrOrders))
        b = 0
        while len(orders) > 0:
            o = orders.pop()
            self.solution[b].append((Sim_annSolver.REST, o))
            self.solution[b].append((Sim_annSolver.CUST, o))
            b += 1
            b = b % self.nrBikers
        for b in range(self.nrBikers):
            self.costOfRoutes[b] = self.calcCostofRoute(self.solution[b])

    def objectiveFunction(self, routeCosts: np.ndarray) -> float:
        return np.max(routeCosts)

    def updateSolution(self, bikerPair: Tuple[int, int],
                       move: Tuple[int, int], insInx: int) -> None:
        b1 = bikerPair[0]
        b2 = bikerPair[1]
        costChange = np.zeros(self.nrBikers)
        if move[0] > -1 and move[1] > -1:
            self.__11operator(self.solution[b1],
                              self.solution[b2], move[0], move[1])

        else:
            if move[1] == -1:
                self.__01operator(self.solution[b1], self.solution[b2],
                                  move[0], insInx)
            else:
                self.__01operator(self.solution[b2], self.solution[b1],
                                  move[1], insInx)
        costChange[b1] = self.__2optProcedure(self.solution[b1])
        costChange[b2] = self.__2optProcedure(self.solution[b2])
        self.costOfRoutes += costChange

    def lambdaInterchange(self) -> List[Set[Tuple[int, int]]]:
        """WARNING: This function assumes R,C pairs.
        This function implements a 1-interchange mechanism for a carrying 
        capacity of 1 with 1 load per order.
        The function returns the a neighbourhood of solutions to the current 
        solution."""
        neighbourhood = []  # type: List[Set[Tuple[int, int]]]
        for pair in self.searchOrder:
            biker1Route = list(self.solution[pair[0]])
            biker2Route = list(self.solution[pair[1]])
            len1        = len(biker1Route)
            stop1       = len1 + 1 if len1 == 0 else len1
            len2        = len(biker2Route)
            stop2       = len2 + 1 if len2 == 0 else len2
            neighbour = set()  # type: Set[Tuple[int, int]]
            for inx1 in range(0, stop1, 2):
                for inx2 in range(0, stop2, 2):
                    if len1 > 0:
                        neighbour.add((inx1, -1))
                    if len2 > 0:
                        neighbour.add((-1, inx2))
                    if len1 > 0 and len2 > 0:
                        neighbour.add((inx1, inx2))
            neighbourhood.append(neighbour)
        return neighbourhood
    
    """WARNING The following functions determine how orders are transfered between 
    bikers. These needs to be changed to account for capacity != 1 and 
    lambda > 1. """

    def __01operator(self, biker1Route: List, biker2Route: List, inx1: int, inx2: int) -> Tuple[List]:
        r = biker1Route.pop(inx1)
        c = biker1Route.pop(inx1)
        biker2Route.insert(inx2, c)
        biker2Route.insert(inx2, r)
        return biker1Route, biker2Route

    def __10operator(self, biker1Route: List, biker2Route: List, inx1: int, inx2: int) -> Tuple[List]:
        return self.__01operator(biker2Route, biker1Route, inx2, inx1)

    def __11operator(self, biker1Route: List, biker2Route: List, inx1: int, inx2: int) -> Tuple[List]:
        r1 = biker1Route.pop(inx1)
        c1 = biker1Route.pop(inx1)
        r2 = biker2Route.pop(inx2)
        c2 = biker2Route.pop(inx2)
        biker2Route.insert(inx2, c1)
        biker2Route.insert(inx2, r1)
        biker1Route.insert(inx1, c2)
        biker1Route.insert(inx1, r2)
        return biker1Route, biker2Route

    def getCostInsDelProcedure(self, bikerPair: Tuple[int, int], move: Tuple[int, int]) -> np.ndarray:
        """WARNING this functions assumes R,C pairs """
        costChanges = np.zeros(self.nrBikers)
        insInx = -1
        if (move[0] % 2 != 0 and move[0] > -1) or \
                (move[1] % 2 != 0 and move[1] > -1):
            warn("Something is wrong! R,C pair not respected in move suggestion!")
        b1 = bikerPair[0]
        b2 = bikerPair[1]
        if move[0] > -1 and move[1] > -1:
            routes = [list(self.solution[b1]), list(self.solution[b2])]
            r = [routes[0].pop(move[0]), routes[1].pop(move[1])]
            c = [routes[0].pop(move[0]), routes[1].pop(move[1])]
            costChanges[b1] = -1 * self.__calcL(routes[0], move[0], r[0], c[0])
            costChanges[b2] = -1 * self.__calcL(routes[1], move[1], r[1], c[1])
            for n1 in range(2):
                n2 = (n1 + 1) % 2
                rest = r[n2]
                cust = c[n2]
                lmin = np.inf
                for i in range(0, len(routes[n1]) + 1, 2):
                    l = self.__calcL(routes[n1], i, rest, cust)
                    if l < lmin:
                        lmin = l
                costChanges[bikerPair[n1]] += lmin
        else:
            if move[1] == -1:
                (von, nach) = (b1, b2)
                inx = move[0]
            else:
                (von, nach) = (b2, b1)
                inx = move[1]
            routeVon = list(self.solution[von])
            routeNach = list(self.solution[nach])
            r = routeVon.pop(inx)
            c = routeVon.pop(inx)
            costChanges[von] = -1 * self.__calcL(routeVon, inx, r, c)
            lmin = np.inf
            for i in range(0, len(routeNach), 2):
                l = self.__calcL(routeNach, i, r, c)
                if l < lmin:
                    lmin = l
                    insInx = i
            costChanges[nach] = lmin
        return costChanges, insInx

    def __calcL(self, route: List[Tuple[int, int]], inx: int,
                rest: Tuple[int, int], cust: Tuple[int, int]) -> float:
        l = 0.0
        if inx > 1:
            order1 = route[inx - 1]
            if order1[0] != 1:
                warn("Some thing is wrong! Trying to insert after R.")
            l += self.__getDistance(self.nodeDicts[order1[0]][order1[1]],
                                    self.nodeDicts[rest[0]][rest[1]])
        l += self.__getDistance(self.nodeDicts[rest[0]][rest[1]],
                                self.nodeDicts[cust[0]][cust[1]])
        if inx < len(route) - 1:
            order2 = route[inx]
            if order2[0] != 0:
                warn("Some thing is wrong! Trying to insert before C.")
            l += self.__getDistance(self.nodeDicts[cust[0]][cust[1]],
                                    self.nodeDicts[order2[0]][order2[1]])
        if inx > 1 and inx < len(route) - 1:
            l -= self.__getDistance(self.nodeDicts[order1[0]][order1[1]],
                                    self.nodeDicts[order2[0]][order2[1]])
        return l

    def __2optProcedure(self, route: List[Tuple[int, int]]) -> float:
        """WARNING: Assumes R,C pairs. WARNING: This function is not tested."""
        foundBetter = True
        bestCost = self.calcCostofRoute(route)
        while foundBetter:
            foundBetter = False
            for i in range(0, len(route) - 1, 2):
                for k in range(i + 1, len(route), 2):
                    newRoute = Sim_annSolver.__2optSwap(route, i, k)
                    newCost = self.calcCostofRoute(newRoute)
                    if newCost < bestCost:
                        route = newRoute
                        bestCost = newCost
                        foundBetter = True
                        break
                if foundBetter:
                    break
        return bestCost

    @staticmethod
    def __2optSwap(route: List[Tuple[int, int]], i: int, k: int) -> List[Tuple[int, int]]:
        """WARNING: This function assumes R,C pairs. """
        if i % 2 != 0 or k % 2 != 1:
            warn("Revers ordering starts on Customer or ends on Restaurant")
        newRoute = []  # type: List[Tuple[int, int]]
        newRoute[0:i] = list(route[0:i])
        interRoute = list(route[i:k + 1])
        interRouteRev = [(-1, -1)] * len(interRoute)
        interRoute = list(route[i:k + 1:2])
        interRoute.reverse()
        interRouteRev[0::2] = interRoute
        interRoute = list(route[i + 1:k + 1:2])
        interRoute.reverse()
        interRouteRev[1::2] = interRoute
        newRoute[i:k + 1] = interRouteRev
        newRoute[k + 1:] = list(route[k + 1:])
        return newRoute

    def checkFeasibility(self, bikerPair: Tuple[int, int], move: Tuple[int, int]) -> bool:
        return True

    def calcCostofRoute(self, route: List[Tuple[int, int]]) -> float:
        """Calclulates the total time cost of a route. """
        cost = 0.0
        for i in range(len(route) - 1):
            node1 = self.nodeDicts[route[i][0]][route[i][1]]
            node2 = self.nodeDicts[route[i + 1][0]][route[i + 1][1]]
            cost += self.__getDistance(node1, node2)
        return cost

    def __getDistance(self, start: Node, goal: Node) -> float:
        # Check matrix
        # Otherwise calc distance and add to matrix
        # return 1.0
        distance = 0.0
        if self.costMatrix[start.get_id(), goal.get_id()] > -1:
            distance = self.costMatrix[start.get_id(), goal.get_id()]
        else:
            distance = pf.a_star_search(self.graph, start, goal)
            self.costMatrix[start.get_id(), goal.get_id()] = distance
            self.costMatrix[goal.get_id(), start.get_id()] = distance
        return distance


if __name__ == "__main__":
    data = Problem(3)
    simulator = Sim_annSolver(data)
    #   simulator.initializeSolution()
    #    simulator.solution[0] = [(0,0), (1,0), (0,1), (1,1)]
    #    simulator.solution[1] = [(0,2), (1,2), (0,3), (1,3)]
    #    simulator.solution[2] = [(0,4), (1,4)]
    #    neighbourhood = simulator.lambdaInterchange()
    #    cost = simulator.getCostInsDelProcedure((0, 2), (0, -1))
    simulator.initializeSolution()
    simulator.initializeCoolingParameters()
    #    for o in neighbourhood:
    #        print(o)
