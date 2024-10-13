import numpy as np
import random
import math

from constant import MODEL_CONSTANT
from edge import Edge

class Node(object):
    def __init__(self, input_weight, output_weight=None, age=1,
                 short_term_error=0, long_term_error=0, insertion_threshold=0,
                 target = None):
        self.input_weight = input_weight
        self.output_weight = output_weight
        self.short_term_error = short_term_error
        self.long_term_error = long_term_error
        self.insertion_threshold = insertion_threshold

        self.age = age

        self.neighbors = []

        self.activate_value = 0

        random_number = random.random()
        self.target = np.array([random_number, 1 - random_number]).T
        random_number = random.random()
        self.output_weight = np.array([random_number, 1 - random_number])

        self.actual_output = None

        # self.output_weight = np.array([random_number, 1 - random_number])
        # self.width_of_gauss = 0
        # self.target =np.array([random_number, 1 - random_number]).T

        # self.age = 1
        # self.inherited_error = 0

        # self.activate_value = 0

        self.quality_measure_for_insertion = random_number

    def get_neighbor_size(self):
        return len(self.neighbors)
    
    def get_neighbors(self):
        return self.neighbors
    
    def add_neighbor(self, neighbor):
        edge = Edge(neighbor)
        self.neighbors.append(edge)

        neighbor.neighbors.append(Edge(self))

    def display_neighbor(self):
        for neighbor in self.neighbors:
            print(neighbor.get_node())

    def is_neighbor(self, node):
        return node in self.get_neighbors()
    
    def delete_neighbor(self, edge):
        self.get_neighbors().remove(edge)

        neighbor = edge.get_node()
        neighbor.get_neighbors().remove(edge)

    def display(self):
        print(self, ": \n", self.input_weight)

    def set_activate_value(self, value):
        self.activate_value = value

    def get_input_distance(self, node):
        return  np.linalg.norm(self.input_weight - node.input_weight)
    
    def get_output_distance(self, node):
        return  np.linalg.norm(self.output_weight - node.output_weight)
    
    def get_quality_measure_for_learning(self):
        return float((self.short_term_error + 1) / (self.long_term_error + 1))
    
    def get_input_learning_rate(self, best_node):
        alpha = self.get_quality_measure_for_learning() / (1 + MODEL_CONSTANT.INPUT_ADAPTION_THRESHOLD) + self.age - 1
        learning_rate = MODEL_CONSTANT.LEARNING_RATE_FOR_BEST_NODE if best_node else MODEL_CONSTANT.LEARNING_RATE_FOR_NEIGHBOR

        if alpha <= 0:
            return 0
        return min(alpha, 1) * learning_rate
        
    def update_input_weight(self, input_node, best_node = False):
        self.input_weight = self.input_weight + self.get_input_learning_rate(best_node) * (input_node.input_weight - self.input_weight)

    def get_width_of_gaussian(self):
        if (self.get_neighbor_size() == 0):
            return 0
        total_distance = 0
        for neighbor in self.get_neighbors():
            total_distance += self.get_input_distance(neighbor.get_node())
        return total_distance / self.get_neighbor_size()
    
    def get_output_learning_rate(self):
        alpha = self.get_quality_measure_for_learning() / (1 + MODEL_CONSTANT.OUTPUT_ADAPTION_THRESHOLD) + self.age - 1

        if (alpha < 0):
            return 0
        return min(1, alpha) * MODEL_CONSTANT.OUTPUT_ADAPTION_LEARNING_RATE
    
    def update_quality_measure_for_insertion(self):
        self.quality_measure_for_insertion = self.long_term_error - self.insertion_threshold * (1 + MODEL_CONSTANT.INSERTION_TOLERANCE)

    def get_local_similarity_output_weight(self):
        if (self.get_neighbor_size() == 0):
            return 0
        total_distance = 0
        for neighbor in self.get_neighbors():
            total_distance += self.get_output_distance(neighbor.get_node())
        return total_distance / self.get_neighbor_size()

    def update_error_counter(self):
        self.long_term_error = math.exp(-1 / MODEL_CONSTANT.TL) * self.long_term_error + (1 - math.exp(-1 / MODEL_CONSTANT.TL)) * np.linalg.norm(self.target - self.actual_output)
        self.short_term_error = math.exp(-1 / MODEL_CONSTANT.TS) * self.long_term_error + (1 - math.exp(-1 / MODEL_CONSTANT.TS)) * np.linalg.norm(self.target - self.actual_output)
    
    def decrease_age_for_winner(self):
        self.age = math.exp(-1 / MODEL_CONSTANT.TY) * self.age

    def decrease_insertion_threshold_for_winner(self):
        alpha = (1 + abs(self.get_quality_measure_for_learning() - 1)) / (1 + MODEL_CONSTANT.INPUT_ADAPTION_THRESHOLD) - 1

        alpha = 0 if alpha < 0 else min(1, alpha)
        self.insertion_threshold = (1 - alpha) * math.exp(-1 / MODEL_CONSTANT.TV) * self.insertion_threshold

    def update_edge_for_winner(self, second):
        if (not self.is_neighbor(second)):
            self.add_neighbor(second)

        for neighbor in self.get_neighbors():
            neighbor.age += 1
            if (neighbor == second):
                neighbor.age = 0
    
    def update_edge(self):
        for neighbor in self.get_neighbors():
            if (self.age > MODEL_CONSTANT.MAXIMUM_EDGE_AGE):
                self.delete_neighbor(neighbor)
            
        
            
        
