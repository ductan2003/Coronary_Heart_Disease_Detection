from node import Node
import numpy as np
import math

class Edge:
    def __init__(self, node, weight=1):
        self.node = node
        self.weight = weight

class Graph(object):
    def __init__(self):
        self.adjacency_list = {}
        # self.graph_size = len(self.graph)

    def add_node(self, node):
        """Adds a new node to the graph."""
        # node = Node(value)
        if node not in self.adjacency_list:
            self.adjacency_list[node] = []
        return node
    
    def add_edge(self, node1, node2, directed=False):
        """Adds an edge between two nodes."""
        # Add the edge from node1 to node2
        self.adjacency_list[node1].append(node2)
        # If undirected, add the reverse edge from node2 to node1
        if not directed:
            self.adjacency_list[node2].append(node1)

    def remove_edge(self, node1, node2):
        self.adjacency_list[node1].remove(node2)
        self.adjacency_list[node2].remove(node1)

    def check_neighbor(self, node1, node2):
        return (node2 in self.adjacency_list[node1]) or (node1 in self.adjacency_list[node2])

    def display(self):
        """Displays the graph."""
        print("Displayed")
        # print(self.adjacency_list)
        for node, neighbors in self.adjacency_list.items():
            print(f"{node.input_weight}: {[neighbor.age for neighbor in neighbors]}")

    def get_graph_size(self):
        return len(self.adjacency_list)
    
    def delete_edge_from_node(self, node):
        for item in self.adjacency_list[node]:
            self.remove_edge(item)

    def find_best_and_second_best_node(self, input_node):
        # Initialize variables to store the best and second-best nodes and their distances
        best_node = None
        second_best_node = None
        best_distance = float('inf')
        second_best_distance = float('inf')

        # Iterate over all nodes in the graph
        for node in self.adjacency_list:
            if node == input_node:
                continue  # Skip comparing the node with itself
            
            # Calculate the distance between the input node and the current node
            distance = input_node.distance(node)
            
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
    
    def get_neighbor_size(self, node):
        return len(self.adjacency_list[node])
    
    def get_width_of_gaussian(self, node):
        if (self.get_neighbor_size(node) == 0):
            return 0
        total_distance = 0
        for neighbor in self.adjacency_list[node]:
            total_distance += node.distance(neighbor)
        return total_distance / self.get_neighbor_size(node)
    
    def activate_node(self, input_node):
        for node in self.adjacency_list:
            width_of_gaussian = self.get_width_of_gaussian(node)
            print(width_of_gaussian)
            if (width_of_gaussian):
                node.activate_value = math.exp(-input_node.distance(node) / (width_of_gaussian * width_of_gaussian))
                # print("get_actual_output", self.get_actual_output().T)
                # print("abc")
                # print(node.get_output_learning_rate())
                # node.output_weight = node.output_weight + node.get_output_learning_rate() * (self.get_actual_output().T )
                # print(node.activate_value)
            else:
                node.activate_value = 0
            print("______")

    def get_actual_output(self):
        activate_values = []
        output_weights = []
        for node in self.adjacency_list:
            activate_values.append(node.activate_value)
            output_weights.append(node.output_weight)
        return np.array(activate_values).dot((np.array(output_weights)))
    
    def update_out_weight(self):
        for node in self.adjacency_list:
            node.output_weight = node.output_weight + node.get_output_learning_rate() * (node.target - self.get_actual_output())
            print(node.output_weight)

    def update_error_counter(self, node):
        print("__________")
        print(node.long_term_error)
        print(node.short_term_error)
        print(node.target)
        # print(node.actual_output)
        node.long_term_error = math.exp(-1 / node.TL) * node.long_term_error + (1 - math.exp(-1 / node.TL)) * np.linalg.norm(node.target - self.get_actual_output())
        node.short_term_error = math.exp(-1 / node.TS) * node.short_term_error + (1 - math.exp(-1 / node.TS)) * np.linalg.norm(node.target - self.get_actual_output())
    
        print(node.long_term_error)
        print(node.short_term_error)

    def update_winner_age(self, node):
        node.age = math.exp(-1 / node.TY) * node.age

    def update_BI(self):
        for node in self.adjacency_list:
            node.update_quality_measure_for_insertion()

    
    def get_q_and_f_node(self):
        q =  max(self.adjacency_list.keys(), key=lambda k: (k.quality_measure_for_insertion - k.age))
        # q = q if (q.quality_measure_for_insertion - q.age) > 0 else None
        # if (q and self.adjacency_list[q]):
        if (True):
            f = max(self.adjacency_list[q], key=lambda k: k.quality_measure_for_insertion)
            if (f):
                is_neighbor = self.check_neighbor(q,f)
                if (is_neighbor):
                    # self.remove_edge(q,f)
                    print((q.output_weight + f.output_weight) / 2)
                    r = Node(input_weight= (q.input_weight + f.input_weight) / 2, 
                            output_weight= (q.output_weight + f.output_weight) / 2, 
                            short_term_error = (q.short_term_error + f.short_term_error) / 2, 
                            long_term_error = (q.long_term_error + f.long_term_error) / 2, 
                            insertion_threshold = (q.insertion_threshold + f.insertion_threshold) / 2, 
                            )
                    # check_insertion = all(x.long_term_error >= x.insertion_threshold * (1 - x.insertion_tolerance) for x in [q,f,r])
                    insert_success = True
                    for item in [q, f, r]:
                        if (item.long_term_error >= item.insertion_threshold * (1 - item.insertion_tolerance)):
                            insert_success = False
                            item.insertion_threshold += item.learning_rate_insertion * (item.long_term_error - item.insertion_threshold * (1 - item.insertion_tolerance))
                    if (insert_success):
                        print("insert success")
                        self.add_node(r)
                        self.remove_edge(q,f)
                        self.add_edge(r,q)
                        self.add_edge(r,f)
                    else:
                        print("insert fail")
                        for item in [q, f, r]:
                            item.long_term_error = item.inherited_error

    def get_average_similarity_input_weight(self, node):
        distance = 0
        for item in self.adjacency_list:
            distance += node.distance(item)
        return distance / self.get_graph_size()

    def get_local_similarity_output_weight(self, node):
        if (self.get_neighbor_size(node) == 0):
            return 0
        total_distance = 0
        for neighbor in self.adjacency_list[node]:
            total_distance += node.distance_output(neighbor)
        return total_distance / self.get_neighbor_size(node)

    def check_deletion_criteria(self):
        min =  float('inf')
        min_item = None
        for item in self.adjacency_list:
            k_deletion = self.get_width_of_gaussian(item) * self.get_local_similarity_output_weight(item) / self.get_average_similarity_input_weight(item)
            if (k_deletion < min):
                min = k_deletion
                min_item = item
        print(min)
        print(min_item)
        if (min_item.deletion_threshold > min and self.get_neighbor_size(min.item) >= 2 and min_item.age > min_item.minimal_age and min_item.get_quality_measure_for_learning() < min_item.suffcient_stabilization):
            print("delete")
        # return 1
