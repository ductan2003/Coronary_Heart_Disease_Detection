import numpy as np
import os
import random

class Edge:
    def __init__(self, node, age = 1):
        self.node = node
        self.age = age

    def get_node(self):
        return self.node

class Node:
    def __init__(self, input_weight, errors=0, label=None):
        self.input_weight = input_weight  # Coordinates of the neuron
        self.errors = errors  # Error accumulated from input data
        self.label = label  # Label associated with the neuron (for classification)
        self.neighbors = []

    def get_neighbor_size(self):
        return len(self.neighbors)
    
    def get_neighbors(self):
        return self.neighbors
    
    def get_neighbor_by_node(self, other_node):
        for neighbor in self.get_neighbors():
            if (neighbor.get_node() == other_node):
                return neighbor
        return
    
    # def add_neighbor(self, neighbor):
    #     edge = Edge(neighbor)
    #     self.neighbors.append(edge)

    #     neighbor.neighbors.append(Edge(self))

    def add_neighbor(self, neighbor):
        edge = Edge(neighbor)
        self.neighbors.append(edge)
        
        # Create a mirrored edge for the neighbor
        mirrored_edge = Edge(self)
        neighbor.neighbors.append(mirrored_edge)
        

    def display_neighbor(self):
        for neighbor in self.neighbors:
            print(neighbor.get_node())

    def is_neighbor(self, node):
        for neighbor in self.get_neighbors():
            if neighbor.get_node() == node:
                return True
        return False
    
    def delete_neighbor_by_edge(self, edge):
        self.get_neighbors().remove(edge)

        neighbor = edge.get_node()
        neighbor.get_neighbors().remove(edge)

    def delete_neighbor_by_node(self, node):
        # Remove the neighbor from this node's list of neighbors
        self.neighbors = [edge for edge in self.neighbors if edge.get_node() != node]

        # Also remove this node from the neighbor's list of neighbors if bidirectional
        node.neighbors = [edge for edge in node.neighbors if edge.get_node() != self]

    def get_input_distance(self, node):
        return  np.linalg.norm(self.input_weight - node.input_weight)
            
    def update_input_weight(self, input_node, learning_rate):
        self.input_weight = self.input_weight + learning_rate * (input_node.input_weight - self.input_weight)

class Graph(object):
    def __init__(self):
        self.graph = []

        np.random.seed(42)
        input = np.random.randint(0, 2, (30, 30))
        new_row = np.zeros((1, 30), dtype=int)
        updated_input_array = np.append(input, new_row, axis=0)
        random_node_1 = Node(input_weight=updated_input_array)
        self.graph.append(random_node_1)

        input = np.random.randint(0, 2, (30, 30))
        new_row = np.zeros((1, 30), dtype=int)
        updated_input_array = np.append(input, new_row, axis=0)
        random_node_2 = Node(input_weight=updated_input_array)
        self.graph.append(random_node_2)

        random_node_1.add_neighbor(random_node_2)

    def add_node(self, node):
        """Adds a new node to the graph."""
        # node = Node(value)
        if node not in self.graph:
            self.graph.append(node)

    def remove_node(self, node):
        self.graph.remove(node)

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

class GNG:
    def __init__(self, dataset_path, max_nodes=1000, winner_lr=0.2, second_lr=0.05, max_edge = 100, theta = 10):
        self.dataset_path = dataset_path
        self.max_nodes = max_nodes
        self.graph = Graph()

        self.winner_lr = winner_lr
        self.second_lr = second_lr
        self.max_edge = max_edge
        self.step = 0
        self.theta = 10
        self.alpha = 0.5

        self.train_error = 0
        self.success_samples = 0

    def fit(self, max_epoch):
        for i in range(max_epoch):
            samples_list = []
            for filename in os.listdir(self.dataset_path):
                if (filename.endswith(".npy")):
                    samples_list.append(filename)
            random.seed(42)
            random.shuffle(samples_list)

            for filename in samples_list:
                self.step += 1
                file_path = os.path.join(self.dataset_path, filename)
                ecg_array = np.load(file_path)

                parts = filename.split('_')
                if len(parts) >= 3:
                    timestamp, label, file_type = parts

                label = int(label)

                input_node = Node(ecg_array, label=label)
                first, second = self.graph.find_best_and_second_best_node(input_node)

                if (label == first.label):
                    self.success_samples += 1
                first.label = label

                for neighbor in first.get_neighbors():
                    neighbor.age += 1
                    neighbor_edge = neighbor.get_node().get_neighbor_by_node(first)
                    neighbor_edge.age += 1

                if (first.is_neighbor(second)):
                        first_edge = first.get_neighbor_by_node(second)
                        first_edge.age = 0
                        second_edge = second.get_neighbor_by_node(first)
                        second_edge.age = 0
                else:
                    first.add_neighbor(second)

                first.errors = first.errors + first.get_input_distance(input_node)

                first.update_input_weight(input_node, self.winner_lr)
                for neighbor in first.get_neighbors():
                    neighbor.get_node().update_input_weight(input_node, self.second_lr)


                if (self.step % self.theta == 0):
                    max_error = 0
                    max_node = None
                    for node in self.graph.graph:
                        if (max_error < node.errors):
                            max_error = max(max_error, node.errors)
                            max_node = node
                        for neighbor in node.get_neighbors():
                            if (neighbor.age > self.max_edge):
                                node.delete_neighbor_by_edge(neighbor)

                    max_error = float('-inf')
                    second_node = None
                    for neighbor in max_node.get_neighbors():
                        neighbor_node = neighbor.get_node()
                        if (max_error < neighbor_node.errors):
                            max_error = max(max_error, neighbor_node.errors)
                            second_node = neighbor_node

                    new_node = Node(input_weight=(max_node.input_weight + second_node.input_weight) / 2, 
                                    label=max_node.label,
                                    errors=(max_node.errors + second_node.errors)/2)                    
                    new_node.add_neighbor(max_node)
                    new_node.add_neighbor(second_node)
                    self.graph.add_node(new_node)
                    max_node.delete_neighbor_by_node(second_node)
                    max_node.errors = self.alpha * max_node.errors
                    second_node.errors = self.alpha * second_node.errors
                

dataset_path = '../LLCS/Disease_dataset/Ex1_Overlap_3000/NumpyData/'
model = GNG(dataset_path=dataset_path, max_nodes=100)
model.fit(max_epoch=1)

print(model.success_samples/model.step)
print(len(model.graph.graph))

