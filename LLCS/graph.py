import math
import numpy as np

from node import Node
from constant import MODEL_CONSTANT

class Graph(object):
    def __init__(self):
        self.log = {}
        self.log["fail"] = 0
        self.capture_data = []
        self.graph = []

        np.random.seed(42)
        input = np.random.randint(0, 2, (30, 30))
        new_row = np.zeros((1, 30), dtype=int)
        updated_input_array = np.append(input, new_row, axis=0)
        output = np.random.uniform(0, 1, (1, 4))
        random_node_1 = Node(input_weight=updated_input_array, output_weight=output, insertion_threshold=1)
        self.graph.append(random_node_1)

        input = np.random.randint(0, 2, (30, 30))
        new_row = np.zeros((1, 30), dtype=int)
        updated_input_array = np.append(input, new_row, axis=0)
        output = np.random.uniform(0, 1, (1, 4))
        random_node_2 = Node(input_weight=updated_input_array, output_weight=output, insertion_threshold=1)
        self.graph.append(random_node_2)

        random_node_1.add_neighbor(random_node_2)
        
        random_node_1.display()
        random_node_2.display()

    def get_graph_size(self):
        return len(self.graph)
    
    def get_graph_edge(self):
        count = 0
        for node in self.graph:
            count += node.get_neighbor_size()

        return count / 2

    def add_node(self, node):
        """Adds a new node to the graph."""
        # node = Node(value)
        if node not in self.graph:
            self.graph.append(node)

    def remove_node(self, node):
        self.graph.remove(node)

    def display(self):
        """Displays the graph."""
        index = 0
        for node in self.graph:
            print(node.display_neighbor())
            index += 1


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
            if distance <= best_distance:
                second_best_distance = best_distance
                second_best_node = best_node
                best_distance = distance
                best_node = node
            elif distance <= second_best_distance:
                second_best_distance = distance
                second_best_node = node

        return best_node, second_best_node
    
    def activate_node(self, input_node):
        for node in self.graph:
            node.update_width_of_gaussian()
            width_of_gaussian = node.get_width_of_gaussian()
            if (width_of_gaussian):
                node.set_activate_value(math.exp(-input_node.get_input_distance(node) / (width_of_gaussian * width_of_gaussian)))
            else:
                node.set_activate_value(0)
            # print("___ACTIVATE DONE___: ", node.activate_value)

    def get_actual_output(self):
        activate_values = []
        output_weights = []
        for node in self.graph:
            activate_values.append(node.activate_value)
            output_weights.append(node.output_weight)
        # print(np.array(activate_values))
        # print(output_weights)
        # print("output by model ", np.array(activate_values).dot(np.concatenate(output_weights, axis=0)))
        return np.array(activate_values).dot(np.concatenate(output_weights, axis=0))

    def update_out_weight(self, input_node):
        input_node.actual_output = self.get_actual_output()
        # print(self.get_actual_output())
        # print(input_node.target)
        # print(input_node.target - input_node.actual_output)
        for node in self.graph:
            # print(node.output_weight)
            node.output_weight = node.output_weight + node.get_output_learning_rate() * (input_node.target - input_node.actual_output)
            # print(node.output_weight)

    def update_BI(self):
        for node in self.graph:
            node.update_quality_measure_for_insertion()

    #TODO
    def get_q_and_f_node(self):
        q = max(self.graph, key=lambda k: (k.quality_measure_for_insertion - k.age))
        # print("q.age", q.age)
        # print("q.quality_measure_for_insertion", q.quality_measure_for_insertion)
        q = q if (q.quality_measure_for_insertion - q.age) > 0 else None
        if (q != None):
            # print("________FOUND Q______________")
            f_edge = max(q.get_neighbors(), key=lambda k: k.get_node().quality_measure_for_insertion)
            f = f_edge.get_node()
            f.display()
            # print("display f")
            if (f):
                # is_neighbor = self.check_neighbor(q,f)
                # if (is_neighbor):
                    # self.remove_edge(q,f)
                # print("q ", q.output_weight)
                # print("f ", f.output_weight)
                # print((q.output_weight + f.output_weight) / 2)
                r = Node(input_weight= (q.input_weight + f.input_weight) / 2, 
                        output_weight= (q.output_weight + f.output_weight) / 2, 
                        short_term_error = (q.short_term_error + f.short_term_error) / 2, 
                        long_term_error = (q.long_term_error + f.long_term_error) / 2, 
                        insertion_threshold = (q.insertion_threshold + f.insertion_threshold) / 2, 
                        inherited_error = (q.inherited_error + f.inherited_error) / 2, 
                        )
                # check_insertion = all(x.long_term_error >= x.insertion_threshold * (1 - x.insertion_tolerance) for x in [q,f,r])
                insert_success = True
                # print("__________q_________", q.display())
                # print("**********************************")
                # print("__________f_________", f.display())
                # print("**********************************")
                # print("__________r_________", r.display())
                # print("**********************************")
                for item in [q, f, r]:
                    if (item.long_term_error >= item.inherited_error * (1 - MODEL_CONSTANT.INSERTION_TOLERANCE)):
                        # print("item.long_term_error", item.long_term_error)
                        # print(item.inherited_error * (1 - MODEL_CONSTANT.INSERTION_TOLERANCE))
                        insert_success = False
                        item.insertion_threshold += MODEL_CONSTANT.INSERTION_LEARNING_RATE * (item.long_term_error - item.insertion_threshold * (1 - MODEL_CONSTANT.INSERTION_TOLERANCE))
                if (insert_success):
                    self.log["success"] = 1000
                    # print("insert success")
                    self.add_node(r)
                    q.delete_neighbor_by_node(f)
                    r.add_neighbor(q)
                    r.add_neighbor(f)
                else:
                    # print("insert fail")
                    if self.log["fail"] != None:
                        self.log["fail"] = self.log["fail"] + 1 
                    else:
                        self.log["fail"] = 0
                    for item in [q, f, r]:
                        # print(item.long_term_error)
                        item.inherited_error = item.long_term_error
                        item.display()

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
        # print(min)
        # print(min_item)
        # print("quality for larning",min_item.get_quality_measure_for_learning())
        # print("neighbor size", min_item.get_neighbor_size())
        # print("age", min_item.age)
        if (MODEL_CONSTANT.DELETION_THRESHOLD > min and min_item.get_neighbor_size() >= 2 and min_item.age < MODEL_CONSTANT.MINIMAL_AGE and min_item.get_quality_measure_for_learning() < MODEL_CONSTANT.SUFFICIENT_STABILIZATION):
            # print("delete")
            # self.log["delete"] = "true"
            self.remove_node(min_item)
        # else:
        #     print("no delete")

    def update_node(self):
        for node in self.graph:
            if (node.get_neighbor_size() == 0):
                self.remove_node(node)
