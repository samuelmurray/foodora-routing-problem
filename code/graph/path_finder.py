""" Functions to find shortest path in our graph """

import math
from typing import List
from graph.node import Node
from graph.graph import Graph
from priority_queue import PriorityQueue


def distance(node1: Node, node2: Node) -> float:
    # This function does not give the graph distance, but the Euclidian distance.
    # That is, it's an underestimate of the travel distance, and therefore suitable as a heuristic.
    x1, y1 = node1.coordinates()
    x2, y2 = node2.coordinates()
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


"""
Courtesy of http://www.redblobgames.com/pathfinding/a-star/implementation.html
Updated to work for our graph class
"""


def a_star_search(graph: Graph, start: Node, goal: Node) -> List[Node]:
    came_from, cost_so_far = __search(graph, start, goal)
    return __reconstruct_path(came_from, start, goal), cost_so_far[goal]


def __search(graph: Graph, start: Node, goal: Node):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next_node in graph.neighbours(current):
            new_cost = cost_so_far[current] + distance(current, next_node)
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + distance(goal, next_node)
#                print("Priority: ", priority)
#                print("next_node: ", next_node.name())
                frontier.put(next_node, priority)
                came_from[next_node] = current

    return came_from, cost_so_far


def __reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()  # optional
    return path
