import numpy as np

from graph import Graph
from node import Node

class LLCS(object):
    def __init__(self, input_adaption_threshold, learning_rate_b, learning_rate_n, lamda = 10):
        self.graph = Graph()
        self.input_adaption_threshold = input_adaption_threshold
        self.learning_rate_b = learning_rate_b
        self.learning_rate_n = learning_rate_n
        self.lamda = lamda

        self.train_data = []
        self.train_data_size = 1000

    # def fit(self):
    #     for i in range(self.train_data_size):
    #         if (i == self.lamda * self.graph.get_graph_size()):


a = LLCS(input_adaption_threshold = 0.001, learning_rate_b = 0.1, learning_rate_n = 0.01)

b = Node(np.array([[1, 2], [3, 4], [5, 6]]))

c = Node(np.array([[2, 1], [3, 7], [2, 3]]))

# print(b.age)
a.graph.add_node(b)

a.graph.add_node(c)
d = Node(np.array([[9, 5], [9, 3], [3, 6]]))
a.graph.add_node(d)
# a.graph.add_node(Node(np.array([[2, 3], [1, 9], [1, 2]])))

a.graph.display()

a.graph.add_edge(b,c)
a.graph.add_edge(b,d)

first, second = a.graph.find_best_and_second_best_node(Node(np.array([[9, 2], [2, 4], [2, 9]])))
first.update_input_weight(np.array([[9, 2], [2, 4], [2, 9]]))
second.update_input_weight(np.array([[9, 2], [2, 4], [2, 9]]), best_node = False)

a.graph.display()

# print(a.graph.get_neighbor_size(b))
# print(a.graph.get_width_of_gaussian(b))
# print(first, second)
a.graph.activate_node(Node(np.array([[9, 2], [2, 4], [2, 9]])))
print(a.graph.get_actual_output())
a.graph.update_out_weight()
a.graph.update_error_counter(first)
a.graph.update_winner_age(first)
print(first.age)
# print(len(a.graph.adjacency_list))
# print(a.graph.adjacency_list[b])
# a.graph.update_BI()
# # a.graph.get_q_and_f_node()
# print(a.graph.get_average_similarity_input_weight(b))
# print(a.graph.get_width_of_gaussian(b))
# print(a.graph.check_deletion_criteria())
# # print(a.graph.check_deletion_criteria())
# print(a.graph.get_actual_output())
