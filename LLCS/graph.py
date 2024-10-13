import math
import numpy as np

from node import Node
from constant import MODEL_CONSTANT

class Graph(object):
    def __init__(self):
        self.graph = []

    def add_node(self, node):
        """Adds a new node to the graph."""
        # node = Node(value)
        if node not in self.graph:
            self.graph.append(node)

    def remove_node(self, node):
        self.graph.remove(node)

    def display(self):
        """Displays the graph."""
        print("Displayed")
        # print(self.adjacency_list)
        for node in self.graph:
            print(node.display())


    def find_best_and_second_best_node(self, input_node):
        # Initialize variables to store the best and second-best nodes and their distances
        best_node = None
        second_best_node = None
        best_distance = float('inf')
        second_best_distance = float('inf')

        # Iterate over all nodes in the graph
        for node in self.graph:
            if node == input_node:
                continue  # Skip comparing the node with itself
            
            # Calculate the distance between the input node and the current node
            distance = input_node.get_input_distance(node)
            
            # Update the best and second-best nodes based on the distance
            if distance < best_distance:
                second_best_distance = best_distance
                second_best_node = best_node
                best_distance = distance
                best_node = node
            elif distance < second_best_distance:
                second_best_distance = distance
                second_best_node = node

        return best_node, second_best_node
    
    def activate_node(self, input_node):
        for node in self.graph:
            width_of_gaussian = node.get_width_of_gaussian()
            if (width_of_gaussian):
                node.set_activate_value(math.exp(-input_node.get_input_distance(node) / (width_of_gaussian * width_of_gaussian)))
            else:
                node.set_activate_value(0)
            print("___ACTIVATE DONE___: ", node.activate_value)

    def get_actual_output(self):
        activate_values = []
        output_weights = []
        for node in self.graph:
            activate_values.append(node.activate_value)
            output_weights.append(node.output_weight)
        return np.array(activate_values).dot((np.array(output_weights)))

    def update_out_weight(self):
        for node in self.graph:
            node.output_weight = node.output_weight + node.get_output_learning_rate() * (node.target - self.get_actual_output())
            print(node.output_weight)

    def update_BI(self):
        print("_______________")
        for node in self.graph:
            node.update_quality_measure_for_insertion()

    #TODO
    def get_q_and_f_node(self):
        q =  max(self.graph, key=lambda k: (k.quality_measure_for_insertion - k.age))
        # q = q if (q.quality_measure_for_insertion - q.age) > 0 else None
        q.display()
        if (True):
            f = max(q.get_neighbors(), key=lambda k: k.get_node().quality_measure_for_insertion).get_node()
            f.display()
            if (f):
                # is_neighbor = self.check_neighbor(q,f)
                # if (is_neighbor):
                    # self.remove_edge(q,f)
                print((q.output_weight + f.output_weight) / 2)
                r = Node(input_weight= (q.input_weight + f.input_weight) / 2, 
                        output_weight= (q.output_weight + f.output_weight) / 2, 
                        short_term_error = (q.short_term_error + f.short_term_error) / 2, 
                        long_term_error = (q.long_term_error + f.long_term_error) / 2, 
                        insertion_threshold = (q.insertion_threshold + f.insertion_threshold) / 2, 
                        )
                r.display()
                # check_insertion = all(x.long_term_error >= x.insertion_threshold * (1 - x.insertion_tolerance) for x in [q,f,r])
                insert_success = True
                for item in [q, f, r]:
                    if (item.long_term_error >= item.insertion_threshold * (1 - MODEL_CONSTANT.INSERTION_TOLERANCE)):
                        insert_success = False
                        item.insertion_threshold += MODEL_CONSTANT.INSERTION_LEARNING_RATE * (item.long_term_error - item.insertion_threshold * (1 - MODEL_CONSTANT.INSERTION_TOLERANCE))
                if (insert_success):
                    print("insert success")
                    self.add_node(r)
                    self.remove_edge(q,f)
                    self.add_edge(r,q)
                    self.add_edge(r,f)
                else:
                    print("insert fail")
                    for item in [q, f, r]:
                        item.inherited_error = item.long_term_error

    def get_average_similarity_input_weight(self):
        distance = 0
        for item in self.graph:
            distance += item.get_width_of_gaussian()
        return distance / len(self.graph)
    
    #TODO
    def check_deletion_criteria(self):
        min =  float('inf')
        min_item = None
        l = self.get_average_similarity_input_weight()
        for item in self.graph:
            k_deletion = item.get_width_of_gaussian() * item.get_local_similarity_output_weight() / l
            if (k_deletion < min):
                min = k_deletion
                min_item = item
        print(min)
        print(min_item)
        print(min_item.get_quality_measure_for_learning())
        if (MODEL_CONSTANT.DELETION_THRESHOLD > min and min_item.get_neighbor_size() >= 2 and min_item.age > MODEL_CONSTANT.MINIMAL_AGE and min_item.get_quality_measure_for_learning() < MODEL_CONSTANT.SUFFICIENT_STABLILIZATION):
            print("delete")
        else:
            print("no delete")

    def update_node(self):
        for node in self.graph:
            if (node.get_neighbor_size() == 0):
                self.remove_node(node)
