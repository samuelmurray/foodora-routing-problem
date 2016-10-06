
from graph import graph


class Problem:
	def __init__(self,graph: graph, init_state: dict, goal_state: dict, bikers: list, restaurants: list, customer_orders: list):
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




