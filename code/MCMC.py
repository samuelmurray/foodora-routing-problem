from graph.node import Node
from graph.graph import Graph
import graph.path_finder as pf
from typing import Dict, List, Tuple, Set
import numpy as np
from warnings import warn
from problem import Problem
import copy

class SimulatedAnnealing:
    """This class contains all the method to solve the foodora VRP. Several 
    functions assume R,C pair. That is, it is assumed the biker goes straight 
    from the restaurant to the customer without pick up of any other food. To 
    generalize the problem, several functions needs to be changed."""
    REST = 0
    CUST = 1
    
    def __init__(self, data: Problem, savePlotData = False) -> None:
        """ Read in the problem data and store of use."""
        # Take the order dict and make a List of tuples. The first
        # index is typ type, "biker", "REST" or "CUST"
        self.bikerStart     = data.bikers()
        self.nrBikers       = len(self.bikerStart)
        self.graph          = data.graph()
        self.nrNodes        = self.graph.node_count()
        self.costMatrix     = (-1 * np.ones((self.nrNodes, self.nrNodes)) +
                               np.eye(self.nrNodes))
        self.nodeDicts      = [dict(), dict()]  # type: List[Dict[int, Node]]
        self.orders         = data.orders()
        
        self.nrOrders       = 0
        for o in range(len(self.orders)):
            self.nodeDicts[0][self.nrOrders] = self.orders[o][0]
            self.nodeDicts[1][self.nrOrders] = self.orders[o][1]
            self.nrOrders += 1

        self.solution       = dict()  # type: Dict[int, List[Tuple[int, int]]]
        self.costOfRoutes   = None
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
        self.bestCost = None
        self.bestCostOfRoutes = None
        self.savePlotData = savePlotData
        if savePlotData:
            self.costData = []
            self.bestCostData = []
            self.tempData = []

    def runSA(self, R=3, zeroCostR = 200, kmax = np.inf) -> None:
        """Runs the entire simulated annealing algorithm. All sub-functions 
        assumes R, C pairs.
        
        R - the number of resets before termination.
        
        zeroCostR - number of delta = 0 solutions allowed to access in a row. 
            This value is used as a quick fix for situations where the algorithm
            gets stuck switching between two solutions with the same cost."""
        # Do initial heuristic solution, only vor variable number of bikers?
        self.initializeSolution()
        # Initialize cooloing schedule parameters
        nrResets = 0
        nrZeroCosts = 0
        self.bestSolution = copy.deepcopy(self.solution)
        self.bestCost = self.objectiveFunction(self.costOfRoutes)
        self.bestCostOfRoutes = copy.deepcopy(self.costOfRoutes)
        self.Tb = -1
        self.initializeCoolingParameters()
        k = 1
        Tk = self.Ts
        if self.savePlotData:
           self.costData.append(self.objectiveFunction(self.costOfRoutes))
           self.tempData.append(float(Tk))
           self.bestCostData.append((self.bestCost, k))
        #Run cycles until R resets
        while nrResets < R and k < kmax:
            foundNewSol = False
            self.searchOrder = np.random.permutation(self.searchOrder)
            # The the lambda-interchange, neighbourhood. lambda = 1 assumes
            neighbourhood = self.lambdaInterchange()  # type: List[Set[Tuple[int, int]]]
            for n in range(len(self.searchOrder)):
                bikerPair = self.searchOrder[n]
                moveSet = set(neighbourhood[n])
                nrMoves = len(moveSet)
                #Break if we are stuck
                if nrZeroCosts > zeroCostR and kmax == np.inf:
                    nrZeroCosts = 0
                    break
                for m in range(nrMoves):
                    move = moveSet.pop()
                    #calculate initial cost change from procedure (a) in Osman
                    costChanges, insInx = self.getCostInsDelProcedure(bikerPair, move)
                    delta = (self.objectiveFunction(self.costOfRoutes + costChanges) -
                             self.objectiveFunction(self.costOfRoutes))
                    theta = np.random.random()
                    if delta == 0.0:
                        nrZeroCosts += 1
                    # Accepence criteria
                    if (delta <= 0) or (np.exp(-delta / Tk) > theta):
                        self.updateSolution(bikerPair, move, insInx)
                        newCost = self.objectiveFunction(self.costOfRoutes)
                        if newCost < self.bestCost:
                            self.bestSolution = copy.deepcopy(self.solution)
                            self.bestCost = float(newCost)
                            self.bestCostOfRoutes = copy.deepcopy(self.costOfRoutes)
                            self.Tb = Tk
                            if self.savePlotData:
                                self.bestCostData.append((newCost, k))
                            nrResets = 0
                            nrZeroCosts = 0
                        else:
                            pass
                        #update temperature, normal update
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
            if not foundNewSol:
                # Update temperature, reset update
                self.Tr = np.amax([self.Tr / 2, self.Tb])
                Tk = self.Tr
                nrResets += 1
            if self.savePlotData:
               self.costData.append(self.objectiveFunction(self.costOfRoutes))
               self.tempData.append(float(Tk))
            k += 1

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

        return copy.deepcopy(self.solution)

    def initializeSolution(self) -> None:
        """Creates and initial feasible solution. Takes all the orders and 
        distributes them roughly equal among the bikers. Every biker goes 
        directly from the restaurant to the customer."""
        orders = list(range(self.nrOrders))
        self.solution       = dict()  # type: Dict[int, List[Tuple[int, int]]]
        for i in range(self.nrBikers):
            self.solution[i] = []
        self.costOfRoutes   = np.zeros(self.nrBikers)
        b = 0
        while len(orders) > 0:
            o = orders.pop()
            self.solution[b].append((SimulatedAnnealing.REST, o))
            self.solution[b].append((SimulatedAnnealing.CUST, o))
            b += 1
            b = b % self.nrBikers
        for b in range(self.nrBikers):
            self.costOfRoutes[b] = self.calcCostofRoute(self.solution[b], b)

    def initializeCoolingParameters(self) -> None:
        """Initialization of cooling schedule parameters. 
        Ts, Tf, Tr, alpha, gamma are initialized here.Thus is done through a 
        first search over the neighbourhood of the initial solution. """
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
                if np.abs(delta) < deltaMin and np.abs(delta) > 0:
                    deltaMin = np.abs(delta)
                if np.abs(delta) > deltaMax:
                    deltaMax = np.abs(delta)
        self.Ts = deltaMax
        self.Tf = deltaMin
        self.Tr = self.Ts
        self.alpha = self.nrOrders * Nfeas
        self.gamma = self.nrOrders
    
    def objectiveFunction(self, routeCosts: np.ndarray) -> float:
        """The objective function. May be amax or sum."""
        return np.amax(routeCosts)

    def updateSolution(self, bikerPair: Tuple[int, int],
                       move: Tuple[int, int], insInx: int) -> None:
        """Updates the solutions with 01, 10 and 11 operators. Finds "optimal"
        (at least improved) TSP route using 2optProcedure."""
        b1 = bikerPair[0]
        b2 = bikerPair[1]
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
        newCost, newRoute = self.__2optProcedure(b1)
        self.solution[b1] = newRoute
        self.costOfRoutes[b1] = newCost
        newCost, newRoute = self.__2optProcedure(b2)
        self.solution[b2] = newRoute
        self.costOfRoutes[b2] = newCost
        
    """WARNING The following functions determine how orders are transfered between 
    bikers. All these functions assumes R,C pairs. """
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

    def lambdaInterchange(self) -> List[Set[Tuple[int, int]]]:
        """WARNING: This function assumes R,C pairs.
        This function implements a 1-interchange mechanism for a carrying 
        capacity of 1 with 1 load per order.
        The function returns the a neighbourhood of solutions to the current 
        solution."""
        neighbourhood = []  # type: List[Set[Tuple[int, int]]]
        for pair in self.searchOrder:
            len1        = len(self.solution[pair[0]])
            stop1       = len1 + 1 if len1 == 0 else len1
            len2        = len(self.solution[pair[1]])
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

    def getCostInsDelProcedure(self, bikerPair: Tuple[int, int], 
                               move: Tuple[int, int]) -> Tuple[np.ndarray, int]:
        """WARNING this functions assumes R,C pairs. This is the (a) cost 
        heuristic from Osman. Depending on the move, the min over l is 
        calculated. That is, a first approixmimation of where to insert the 
        orders in the new route, and the cost of doing this."""
        costChanges = np.zeros(self.nrBikers)
        insInx = -1
        if (move[0] % 2 != 0 and move[0] > -1) or \
                (move[1] % 2 != 0 and move[1] > -1):
            warn("Something is wrong! R,C pair not respected in move suggestion!")
        b1 = bikerPair[0]
        b2 = bikerPair[1]
        if move[0] > -1 and move[1] > -1:
            routes = [copy.deepcopy(self.solution[b1]), 
                                    copy.deepcopy(self.solution[b2])]
            r = [routes[0].pop(move[0]), routes[1].pop(move[1])]
            c = [routes[0].pop(move[0]), routes[1].pop(move[1])]
            costChanges[b1] = -1 * self.calcL(routes[0], b1, move[0], r[0], c[0])
            costChanges[b2] = -1 * self.calcL(routes[1], b2, move[1], r[1], c[1])
            for n1 in range(2):
                n2 = (n1 + 1) % 2
                rest = r[n2]
                cust = c[n2]
                lmin = np.inf
                for i in range(0, len(routes[n1]) + 1, 2):
                    l = self.calcL(routes[n1], bikerPair[n1], i, rest, cust)
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
            routeVon = copy.deepcopy(self.solution[von])
            routeNach = copy.deepcopy(self.solution[nach])
            r = routeVon.pop(inx)
            c = routeVon.pop(inx)
            costChanges[von] = -1 * self.calcL(routeVon, von, inx, r, c)
            lmin = np.inf
            for i in range(0, len(routeNach) + 1, 2):
                l = self.calcL(routeNach, nach, i, r, c)
                if l < lmin:
                    lmin = l
                    insInx = i
            costChanges[nach] = lmin
        return costChanges, insInx

    def calcL(self, route: List[Tuple[int, int]], bikerNr: int, inx: int,
                rest: Tuple[int, int], cust: Tuple[int, int]) -> float:
        """Calculates the l value from procedure (a) in Osman."""
        l = 0.0
        restNode = self.nodeDicts[rest[0]][rest[1]]
        custNode = self.nodeDicts[cust[0]][cust[1]]
        if inx > 1:
            order1 = route[inx - 1]
            custNodePre = self.nodeDicts[order1[0]][order1[1]]
            if order1[0] != 1:
                warn("Some thing is wrong! Trying to insert after R.")
            #Add distance from last customer to new restaurant
            l += self.__getDistance(custNodePre, restNode)
        else:
            if inx == 1:
                #sanity check
                warn("WHY IS inx==1 ?!! can't add at a customer index (odd index)")
            #If added as first order, add distance from start noce
            l += self.__getDistance(self.bikerStart[bikerNr], restNode)
        # Add order internal distance
        l += self.__getDistance(restNode, custNode)
        if inx < len(route) - 1:
            order2 = route[inx]
            restNodePost = self.nodeDicts[order2[0]][order2[1]]
            if order2[0] != 0:
                warn("Some thing is wrong! Trying to insert before C.")
            l += self.__getDistance(custNode, restNodePost)
        if len(route) > 0 and inx == 0:
            #if insterted as 0 and not empty route, remove distance to previous first rest
            l -= self.__getDistance(self.bikerStart[bikerNr], restNodePost)
        elif inx > 1 and inx < len(route) - 1:
            l -= self.__getDistance(custNodePre, restNodePost)
        else:
            pass
        return l

    def __2optProcedure(self, bikerNr: int) -> float:
        """WARNING: Assumes R,C pairs. Applies the 2-opt procedure to the route
        of bikerNr. Swaps pairs of edges in the route until no improvement can
        be made. This is procedute (b) in Osman."""
        foundBetter = True
        route = copy.deepcopy(self.solution[bikerNr])
        bestCost = self.calcCostofRoute(route, bikerNr)
        while foundBetter:
            foundBetter = False
            for i in range(0, len(route) - 1, 2):
                for k in range(i + 1, len(route), 2):
                    newRoute = SimulatedAnnealing.__2optSwap(route, i, k)
                    newCost = self.calcCostofRoute(newRoute, bikerNr)
                    if newCost < bestCost:
                        route = newRoute
                        bestCost = newCost
                        foundBetter = True
                        break
                if foundBetter:
                    break
        return bestCost, route

    @staticmethod
    def __2optSwap(route: List[Tuple[int, int]], i: int, k: int) -> List[Tuple[int, int]]:
        """WARNING: This function assumes R,C pairs. Performs the 2-opt swap 
        for the 2-opt procedure."""
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

    def checkFeasibility(self, bikerPair: Tuple[int, int], 
                         move: Tuple[int, int]) -> bool:
        """Not used, but could be used to implement new constraints."""
        return True

    def calcCostofRoute(self, route: List[Tuple[int, int]], bikerNr: int) -> float:
        """Calclulates the total time cost of a route. """
        if len(route) < 1:
            return 0.0
        startNode = self.nodeDicts[route[0][0]][route[0][1]]
        cost = self.__getDistance(self.bikerStart[bikerNr], startNode)
        for i in range(len(route) - 1):
            node1 = self.nodeDicts[route[i][0]][route[i][1]]
            node2 = self.nodeDicts[route[i + 1][0]][route[i + 1][1]]
            cost += self.__getDistance(node1, node2)
        return cost

    def __getDistance(self, start: Node, goal: Node) -> float:
        """Calculates the cost of travelning between 2 nodes. Once done, the 
        value is stored in a matix so A* not needed to be run again.""" 
        distance = 0.0
        if self.costMatrix[start.get_id(), goal.get_id()] > -1:
            distance = self.costMatrix[start.get_id(), goal.get_id()]
        else:
            seq, distance = pf.a_star_search(self.graph, start, goal)
            self.costMatrix[start.get_id(), goal.get_id()] = distance
            self.costMatrix[goal.get_id(), start.get_id()] = distance
        return distance