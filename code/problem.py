
from graph import graph


class Problem:
	def __init__(self.graph: graph, init_state: dict, goal_state: dict, bikers: list, restaurants: list, customer_orders: list):
		self.graph = graph
		self.init_state = init_state
		self.goal_state = goal_state
		self.bikers = bikers
		self.restaurants = restaurants
		self.customer_orders = customer_orders

	
	def get_graph():
		return self.graph

	def get_init_state():
		return self.init_state

	def get_goal_state():
		return self.goal_state

	def get_bikers():
		return self.bikers

	def get_restaurants():
		return self.restaurants

	def get_customer_orders():
		return self.customer_orders

	def make_pddl():
		with open('pddl_init.ppdl', 'w') as out:
		    out.write('(define (problem ppdl_init)\n (:domain foodora_domain)\n (:objects ')
		    for c in self.customer_orders.keys():
		        out.write(c + ' ')
		    out.write('- customer\n')
		    for r in self.restaurants:
		        out.write(r + ' ')
		    out.write('- restaurant\n')
		    for b in self.bikers:
		        out.write(b + ' ')
		    out.write('- biker\n')
		    for n in self.graph.nodes():
		        out.write(n.name() + ' ')
		        out.write('- node)\n')

		    out.write('(:init\n')

		    #Improve when I understand structure on init_state

		    for c in self.customer_orders.keys():
				out.write('at-c ' + c)
		    for r in self.restaurants:
				out.write('at-r ' + r)
		    for e in self.graph.edges():
		    	out.write('edge' + e[0] + ' ' + e[1] + '\n')

		    #out.write('rGotFoodFor ')
		    
		    for b in self.bikers:
		        out.write('notHavFood ' + b + '\n')
		    out.write('- biker\n')

		    out.write('(:goal (and')
		    for c in self.customer_orders.keys():
				out.write(' (gotFood ' + c + ')')
			out.write('))\n)')






