from typing import Dict, Tuple
from graph.graph import Graph
from graph.node import Node
import graph.path_finder as pf


class Problem:
    def __init__(self, graph: Graph, bikers: Dict[int, Node], orders: Dict[int, Tuple[Node, Node]]):
        self.__graph = graph
        self.__bikers = bikers
        self.__orders = orders
        self.__restaurants = self.make_restaurants_list(self.__orders)
        self.__customers = self.make_customer_list(self.__orders)

    def graph(self):
        return self.__graph

    def bikers(self):
        return self.__bikers

    def orders(self):
        return self.__orders

    def restaurants(self):
        return self.__restaurants

    def customers(self):
        return self.__customers

    @staticmethod
    def make_restaurants_list(orders: Dict[int, Tuple[Node, Node]]):
        restaurants = {}
        for i, key in enumerate(orders):
            restaurants[i] = orders[key][0]
        return restaurants

    @staticmethod
    def make_customer_list(orders: Dict[int, Tuple[Node, Node]]):
        customers = {}
        for i, key in enumerate(orders.keys()):
            customers[i] = orders[key][1]
        return customers

    def make_pddl(self):
        with open('data/pddl_init.ppdl', 'w') as out:
            out.write('(define (problem test_problem_cost)\n\t(:domain foodora_cost_domain)\n\t(:objects ')
            for c in self.__customers.keys():
                out.write(str(c) + ' ')
            out.write('- customer\n\t\t\t\t\t')
            for r in self.__restaurants.keys():
                out.write(str(r) + ' ')
            out.write('- restaurant\n\t\t\t\t\t')
            for b in self.__bikers.keys():
                out.write(str(b) + ' ')
            out.write('- biker\n\t\t\t\t\t')
            for n in self.__graph.nodes():
                out.write(n.name() + ' ')
                out.write('- node)\n')

            out.write('\t(:init\n (= (total-cost) 0)\n')

            for c_id, c_n in self.__customers.items():
                out.write('\t(at-c ' + str(c_id) + ' ' + c_n.name() + ')\n')
            for r_id, r_n in self.__restaurants.items():
                out.write('\t(at-r ' + str(r_id) + ' ' + r_n.name() + ')\n')
            for e in self.__graph.edges():
                out.write('\t(edge ' + e[0].name() + ' ' + e[1].name() + ')\n')

            for e2 in self.__graph.edges():
                out.write('\t(= distance ' + e2[0].name() + ' ' + e2[1].name() + ') ' + str(pf.distance(e2[0], e2[1])) + ')\n')

            for b_id, b_n in self.__bikers.items():
                out.write('\t(at-b' + str(b_id) + ' ' + b_n.name() + ')\n')

                for o in self.__orders.values():
                    out.write('\t(rGotFoodFor ' + o[0].name() + ' ' + o[1].name() + ')\n')

                for b2 in self.__bikers.keys():
                    out.write('\t(notHaveFood ' + str(b2) + ')\n')
                out.write(')\n')

                out.write('\t(:goal (and')
                for c2 in self.__customers.keys():
                    out.write(' (gotFood ' + str(c2) + ')')
                out.write('))\n\n\t(:metric minimize (total-cost))\n)')

        '''TODO? def read_pddl_solution():
        '''
