"""
Courtesy of http://www.redblobgames.com/pathfinding/a-star/implementation.html
"""

import heapq


class PriorityQueue:
    def __init__(self) -> None:
        self.elements = []

    def empty(self) -> bool:
        return len(self.elements) == 0

    def put(self, item, priority) -> None:
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]
