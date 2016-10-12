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
        if len([element for element in self.elements if element[0] == priority]) == 0:
            heapq.heappush(self.elements, (priority, item))
        else:
            # Identical values can not be in the heap. Decrement the value by epsilon.
            self.put(item, priority - 0.00001)

    def get(self):
        return heapq.heappop(self.elements)[1]
