""" A data class containing all the information about the problem, initial conditions etc. """

from typing import Dict, Tuple
from graph.graph import Graph
from graph.node import Node
import graph.path_finder as pf


class Problem:
    def __init__(self, graph: Graph, bikers: Dict[int, Node], orders: Dict[int, Tuple[Node, Node]]):
        self.__graph = graph
        self.__bikers = bikers
        self.__orders = orders

    def graph(self):
        return self.__graph

    def bikers(self):
        return self.__bikers

    def orders(self):
        return self.__orders

    def make_pddl(self):
        with open('data/pddl_init.ppdl', 'w') as out:
            out.write('(define (problem test_problem_cost)\n\t(:domain foodora_cost_domain)\n\t(:objects\t')
            for c in self.__orders.keys():
                out.write('c' + str(c) + ' ')
            out.write('- customer\n\t\t\t\t')
            for r in self.__orders.keys():
                out.write('r' + str(r) + ' ')
            out.write('- restaurant\n\t\t\t\t')
            for b in self.__bikers.keys():
                out.write('b' + str(b) + ' ')
            out.write('- biker\n\t\t\t\t')
            for n in self.__graph.nodes():
                out.write(n.name() + ' ')
            out.write('- node)\n')

            out.write('\t(:init\n\t(= (total-cost) 0)\n')

            for key in self.__orders.keys():
                out.write('\t(at-c ' + 'c' + str(key) + ' ' + self.__orders[key][1].name() + ')\n')
            for key in self.__orders.keys():
                out.write('\t(at-r ' + 'r' + str(key) + ' ' + self.__orders[key][0].name() + ')\n')
            for e in self.__graph.edges():
                out.write('\t(edge ' + e[0].name() + ' ' + e[1].name() + ')\n')

            for e2 in self.__graph.edges():
                out.write('\t(= (distance ' + e2[0].name() + ' ' + e2[1].name() + ') ' + str(pf.distance(e2[0], e2[1])) + ')\n')

            for b_id, b_n in self.__bikers.items():
                out.write('\t(at-b b' + str(b_id) + ' ' + b_n.name() + ')\n')

            for key in self.__orders.keys():
                out.write('\t(rGotFoodFor r' + str(key) + ' c' + str(key) + ')\n')

            for b2 in self.__bikers.keys():
                out.write('\t(notHaveFood b' + str(b2) + ')\n')
            out.write('\t)\n')

            out.write('\t(:goal (and')
            for key in self.__orders.keys():
                out.write(' (gotFood c' + str(key) + ')')
            out.write('))\n\n\t(:metric minimize (total-cost))\n\t)')
