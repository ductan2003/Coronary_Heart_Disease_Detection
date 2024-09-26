import numpy as np
import random
import math

class Node(object):
    def __init__(self, input_weight, output_weight = None, short_term_error = 0, long_term_error = 0, insertion_threshold = 0,
                 input_adaption_threshold = 0.001, 
                 learning_rate_b = 0.1, learning_rate_n = 0.01, learning_rate_insertion = 0.1,
                 output_apdation_threshold = -0.05, output_adaption_learning_rate = 0.1, 
                 insertion_tolerance = 0.01,
                 deletion_threshold = 0.01):
        self.input_weight = input_weight

        random_number = random.random()
        print(output_weight)
        # self.output_weight = output_weight if output_weight.any() else np.array([random_number, 1 - random_number]).T
        self.output_weight = np.array([random_number, 1 - random_number])
        # self.output_weight = np.array([1, 0]).T
        self.width_of_gauss = 0
        self.target =np.array([random_number, 1 - random_number]).T
        print("AFS", self.target)
        print("output_weight", self.output_weight)

        self.age = 1
        self.short_term_error = short_term_error
        self.long_term_error = random_number
        self.insertion_threshold = insertion_threshold
        self.inherited_error = 0

        self.input_apdation_threshold = input_adaption_threshold
        self.learning_rate_b = learning_rate_b
        self.learning_rate_n = learning_rate_n

        self.output_apdation_threshold = output_apdation_threshold
        self.output_adaption_learning_rate = output_adaption_learning_rate
        self.activate_value = 0

        self.quality_measure_for_insertion = random_number
        self.insertion_tolerance = insertion_tolerance
        self.learning_rate_insertion = learning_rate_insertion
        self.deletion_threshold = deletion_threshold
        self.minimal_age = 0.01
        self.suffcient_stabilization = 0.01
        self.TL = 100
        self.TS = 10
        self.TY = 100
        self.TV = 100
        self.actual_output = None


    def distance(self, node):
        return  np.linalg.norm(self.input_weight - node.input_weight)
    
    def distance_output(self, node):
        return  np.linalg.norm(self.output_weight - node.output_weight)
    
    def get_quality_measure_for_learning(self):
        return float((self.short_term_error + 1) / (self.long_term_error + 1))
    
    def get_input_learning_rate(self, best_node = True):
        alpha = self.get_quality_measure_for_learning() / (1 + self.input_apdation_threshold) + self.age - 1
        learning_rate = self.learning_rate_b if best_node else self.learning_rate_n

        if (alpha < 0):
            return 0
        elif (alpha < 1):
            return alpha * learning_rate
        else:
            return learning_rate
        
    def update_input_weight(self, input_vector, best_node = True):
        self.input_weight = self.input_weight + self.get_input_learning_rate(best_node) * (input_vector - self.input_weight)
        
    def get_output_learning_rate(self):
        alpha = self.get_quality_measure_for_learning() / (1 + self.output_apdation_threshold) + self.age - 1

        if (alpha < 0):
            return 0
        elif (alpha < 1):
            return alpha * self.output_adaption_learning_rate
        else:
            return self.output_adaption_learning_rate

    def update_quality_measure_for_insertion(self):
        self.quality_measure_for_insertion = self.long_term_error - self.insertion_threshold * (1 + self.insertion_tolerance)
        print(self.quality_measure_for_insertion - self.age)

    # def update_error_counter(self):
    #     print("__________")
    #     print(self.long_term_error)
    #     print(self.short_term_error)
    #     print(self.target)
    #     print(self.actual_output)
    #     self.long_term_error = math.exp(-1 / self.TL) * self.long_term_error + (1 - math.exp(-1 / self.TL)) * (self.target - self.actual_output)
    #     self.short_term_error = math.exp(-1 / self.TS) * self.long_term_error + (1 - math.exp(-1 / self.TS)) * (self.target - self.actual_output)