import numpy as np

from graph import Graph
from node import Node


class LLCS(object):
    def __init__(self):
        self.graph = Graph()



model = LLCS()

a = Node(np.array([[1, 2], [3, 4], [5, 6]]))
b = Node(np.array([[1, 3], [8, 2], [8, 1]]))
c = Node(np.array([[2, 1], [3, 7], [2, 3]]))
d = Node(np.array([[9, 5], [9, 3], [3, 6]]))

model.graph.add_node(a)
a.display()
model.graph.add_node(b)
b.display()
model.graph.add_node(c)
c.display()
model.graph.add_node(d)
d.display()
print("_______________")
a.add_neighbor(b)
d.add_neighbor(a)

# model.graph.display()

input_node = Node(np.array([[9, 2], [2, 4], [2, 9]]))
first, second = model.graph.find_best_and_second_best_node(input_node)

first.update_input_weight(input_node, best_node = True)
second.update_input_weight(input_node)

model.graph.activate_node(input_node)

print(model.graph.get_actual_output())

model.graph.update_out_weight()

model.graph.update_BI()
model.graph.get_q_and_f_node()

model.graph.check_deletion_criteria()
first.actual_output = model.graph.get_actual_output()
first.update_error_counter()
