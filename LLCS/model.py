import numpy as np
import pandas as pd
import os
import json

from graph import Graph
from node import Node
from constant import MODEL_CONSTANT

import numpy as np

def softmax(array):
    # Subtract the maximum value from the array for numerical stability
    exp_values = np.exp(array - np.max(array))
    return exp_values / np.sum(exp_values)

class LLCS(object):
    def __init__(self):
        self.graph = Graph()
        self.step = 0
        self.total_mse = 0
        self.mse = 0

    def fit(self):
        datasetPath = "./Dataset3_Official/NumpyData_3031/train/"

        t_ins = 0
        index = 0
        for filename in os.listdir(datasetPath):
            print("Index", index)
            if (filename.endswith(".npy")):
                file_path = os.path.join(datasetPath, filename)
                ecg_array = np.load(file_path)

                parts = filename.split('_')
                if len(parts) >= 3:
                    timestamp, label, file_type = parts

                label = int(label)
                
                output = np.zeros((1, 6), dtype=int)
                if 0 <= label < 6:
                    # Set the specified index to 1
                    output[0, label] = 1
                else:
                    print("Invalid index. Please provide a value between 0 and 5.")
            
                input_node = Node(ecg_array, target=output)

                # 1.1. find best and second match
                first, second = model.graph.find_best_and_second_best_node(input_node)

                # update quality measure for learning
                first.update_input_weight(input_node, best_node=True)
                for neighbor in first.get_neighbors():
                    neighbor.get_node().update_input_weight(input_node, best_node=False)

                self.graph.activate_node(input_node)
                self.graph.update_out_weight(input_node)

                self.step += 1
                self.total_mse += np.mean((input_node.actual_output - input_node.target) ** 2)
                self.mse = self.total_mse / self.step

                t_ins += 1
                index += 1
                
                if (t_ins == MODEL_CONSTANT.THETA * self.graph.get_graph_size()):
                    print("STEP", index)
                    self.graph.log[index] = "insert"
                    self.graph.update_BI()
                    self.graph.get_q_and_f_node()
                    t_ins = 0

                self.graph.check_deletion_criteria()

                first.update_error_counter(input_node)
                first.decrease_age_for_winner()
                first.decrease_insertion_threshold_for_winner()
                first.update_edge_for_winner(second)

                self.graph.update_node() 
                print("*****************")

                if (index % MODEL_CONSTANT.CAPTURE_TIME == 0):
                    data = {
                        "index": index,
                        "num_nodes": self.graph.get_graph_size(),
                        "edge": self.graph.get_graph_edge(),
                        "mse": self.mse
                    }
                    self.graph.capture_data.append(data)
            # if (index == 500):
            #     break
        # self.graph.display()
        
    def evaluate(self):
        count = 0
        datasetPath = "./Dataset3_Official/NumpyData/val/"

        index = 0
        for filename in os.listdir(datasetPath):
            if (filename.endswith(".npy")):
                index += 1
                file_path = os.path.join(datasetPath, filename)
                ecg_array = np.load(file_path)

                parts = filename.split('_')
                if len(parts) >= 3:
                    timestamp, label, file_type = parts

                # print("label", label)
                label = int(label)

                if label == 0:
                    output = np.array([[1, 0]])
                else:
                    output = np.array([[0, 1]])
            
                input_node = Node(ecg_array, target=output)
                model.graph.activate_node(input_node=input_node)

                output = model.graph.get_actual_output()
                softmax_values = softmax(output)

                print("softmax_values", softmax_values)

                if (softmax_values[1] > softmax_values[0]):
                    detect = 1
                else:
                    detect = 0
                
                if (detect == label):
                    count += 1
        
        print("Total ",count)


model = LLCS()
# model.graph.display()
model.fit()

print(model.graph.log)

from datetime import datetime

# Current timestamp as a float
timestamp = datetime.now().timestamp()

# Convert to string
timestamp_str = str(timestamp)

with open("./Dataset3_" + timestamp_str + ".json", 'w') as json_file:
    json.dump(model.graph.capture_data, json_file, indent=4)  # indent=4 for pretty-printing

for node in model.graph.graph:
    print(node.get_neighbor_size())

# model.evaluate()
