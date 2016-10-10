from typing import Dict, Tuple
from graph.graph import Graph
from graph.node import Node
import graph.path_finder as pf


class Problem:
    def __init__(self, graph: Graph, bikers: Dict[int, Node], orders: Dict[int, Tuple[Node, Node]]):
        self.graph = graph
        self.bikers = bikers
        self.orders = orders
        self.restaurants = self.make_restaurants_list(self.orders)
        self.customers = self.make_customer_list(self.orders)

    def get_graph(self):
        return self.graph

    def get_bikers(self):
        return self.bikers

    def get_restaurants(self):
        return self.restaurants

    def get_customer_orders(self):
        return self.orders

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
        with open('pddl_init.ppdl', 'w') as out:
            out.write('(define (problem test_problem_cost)\n\t(:domain foodora_cost_domain)\n\t(:objects ')
            for c in self.customers.keys():
                out.write(str(c) + ' ')
            out.write('- customer\n\t\t\t\t\t')
            for r in self.restaurants.keys():
                out.write(str(r) + ' ')
            out.write('- restaurant\n\t\t\t\t\t')
            for b in self.bikers.keys():
                out.write(str(b) + ' ')
            out.write('- biker\n\t\t\t\t\t')
            for n in self.graph.nodes():
                out.write(n.name() + ' ')
                out.write('- node)\n')

            out.write('\t(:init\n (= (total-cost) 0)\n')

            for c_id, c_n in self.customers.items():
                out.write('\t(at-c ' + str(c_id) + ' ' + str(c_n) + ')\n')
            for r_id, r_n in self.restaurants.items():
                out.write('\t(at-r ' + str(r_id) + ' ' + str(r_n) + ')\n')
            for e in self.graph.edges():
                out.write('\t(edge' + e[0] + ' ' + e[1] + ')\n')

            for e2 in self.graph.edges():
                out.write('\t(= distance ' + e2[0] + ' ' + e2[1] + ') ' + pf.distance(e2[0], e2[1]) + ')\n')

            for b_id, b_n in self.bikers.values():
                out.write('\t(at-b' + b_id + ' ' + b_n + ')\n')

                for o in self.orders.values():
                    out.write('\t(rGotFoodFor ' + o[0] + ' ' + o[1] + ')\n')

                for b2 in self.bikers.keys():
                    out.write('\t(notHaveFood ' + b2 + ')\n')
                out.write(')\n')

                out.write('\t(:goal (and')
                for c2 in self.customers.keys():
                    out.write(' (gotFood ' + c2 + ')')
                out.write('))\n\n\t(:metric minimize (total-cost))\n)')

        '''TODO? def read_pddl_solution():
        '''
