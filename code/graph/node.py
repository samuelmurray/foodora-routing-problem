"""
Class representing nodes in the graph.
Each node has coordinates which can be used to calculate length of edges.
"""

from typing import List, Dict, Tuple


class Node:
    __nodes = []  # type: List[Node]
    __name_to_node = {}  # type: Dict[str, Node]

    def __init__(self, x: float, y: float, name: str = None) -> None:
        self.__x = x
        self.__y = y
        self.__id = len(Node.__nodes)
        Node.__nodes.append(self)
        if name is not None and name not in self.__name_to_node:
            self.__name = name
            self.__name_to_node[name] = self

    def coordinates(self) -> Tuple[float, float]:
        return self.__x, self.__y

    def name(self) -> str:
        return self.__name
        
    def get_id(self):
        return self.__id

    @staticmethod
    def node_by_name(name: str):
        return Node.__name_to_node[name]

    @staticmethod
    def node_by_id(id: int):
        return Node.__nodes[id] if len(Node.__nodes) >= id else None

    def __str__(self):
        return "[id: {id}, name: {name}, x: {x}, y: {y}]".format(id=self.__id, name=self.__name, x=self.__x, y=self.__y)
