from typing import Dict, List, Iterable, Tuple
from graph.graph import Graph
import graph.path_finder as pf


class Problem:
	def __init__(self, graph: Graph, bikers: dict, orders: dict):
		self.graph = graph
		self.bikers = bikers
		self.orders = orders # Dict[int, Tuple[Node, Node]]
		self.restaurants = make_restaurants_list(self.orders)
		self.customers = make_customer_list(self.orders)

	
	def get_graph():
		return self.graph

	def get_goal_state():
		return self.goal_state

	def get_bikers():
		return self.bikers

	def get_restaurants():
		return self.restaurants

	def get_customer_orders():
		return self.orders

	def make_customer_list(orders: dict):
		customers = {}
		index_cust = 1
		for tup in orders:
			customers[index_cust] = tup[1];
			index_cust += 1

	def make_restaurants_list(orders: dict):
		restaurants = {}
		index_rest = 1
		for tup in orders:
			restaurants[index_rest] = tup[0];
			index_rest += 1


	def make_pddl():

		with open('pddl_init.ppdl', 'w') as out:
		    out.write('(define (problem test_problem_cost)\n\t(:domain foodora_cost_domain)\n\t(:objects ')
		    for c in self.customers.keys():
		        out.write(c[1] + ' ')
		    out.write('- customer\n\t\t\t\t\t')
		    for r in self.restaurants.keys():
		        out.write(r[0] + ' ')
		    out.write('- restaurant\n\t\t\t\t\t')
		    for b in self.bikers.keys():
		        out.write(b + ' ')
		    out.write('- biker\n\t\t\t\t\t')
		    for n in self.graph.nodes(): 
		        out.write(n.name() + ' ')
		        out.write('- node)\n')

		    out.write('\t(:init\n (= (total-cost) 0)\n')

		    for c_id, c_n in self.customers.items():
				out.write('\t(at-c ' + c_id + ' ' + c_n + ')\n')
		    for r_id, r_n in self.restaurants.items():
				out.write('\t(at-r ' + r_id + ' ' + r_n + ')\n')
		    for e in self.graph.edges():
		    	out.write('\t(edge' + e[0] + ' ' + e[1] + ')\n')

		    for e2 in self.graph.edges():
		    	out.write('\t(= distance ' + e2[0] + ' ' + e2[1] + ') ' + pf.distance(e2[0], e2[1]) + ')\n')

		    for b_id, b_n in self.bikers.values():
		        out.write('\t(at-b' + b_id + ' ' + b_n ')\n')

		    for o in self.orders.values():
		    	out.write('\t(rGotFoodFor ' + o[0] + ' ' + o[1] + ')\n')

		    for b2 in self.bikers.keys():
		        out.write('\t(notHaveFood ' + b3 + ')\n')
		    out.write(')\n')

		    out.write('\t(:goal (and')
		    for c2 in self.customers.keys():
				out.write(' (gotFood ' + c2 + ')')
			out.write('))\n\n\t(:metric minimize (total-cost))\n)')

	'''TODO? def read_pddl_solution():
	'''

