import numpy as np
import pandas as pd
import os
import json
from datetime import datetime


from graph import Graph
from node import Node
from constant import MODEL_CONSTANT, DATASET

import numpy as np

def softmax(array):
    # Subtract the maximum value from the array for numerical stability
    exp_values = np.exp(array - np.max(array))
    softmax_value = exp_values / np.sum(exp_values)

    max_value = np.max(softmax_value)

    # Replace maximum with 1 and others with 0
    softmax_value = np.where(softmax_value == max_value, 1, 0)

    return softmax_value.reshape(1, -1)

class LLCS(object):
    def __init__(self, dataset_name, dataset_log_path):
        self.graph = Graph()
        self.dataset_name = dataset_name
        self.step = 0

        self.total_mse = 0

        self.true_detect = 0

        # Current timestamp
        timestamp_str = timestamp_str = datetime.now().strftime("%y-%m-%d-%H:%M:%S")

        self.dataset_log_path = dataset_log_path + timestamp_str + "/"
        os.makedirs(self.dataset_log_path , exist_ok=True)

    def export_log(self):
        with open(self.dataset_log_path + "log_until_" + str(self.step) + ".json", 'w') as json_file:
            json.dump(self.graph.capture_data, json_file, indent=4)  # indent=4 for pretty-printing
            
        self.graph.capture_data = []

    def export_parameter(self):
        model_constants = {attr: value for attr, value in MODEL_CONSTANT.__dict__.items() if not attr.startswith('__')}

        with open(self.dataset_log_path + "parameters_log.json", 'w') as json_file:
            json.dump(model_constants, json_file, indent=4)  # indent=4 for pretty-printing

    def capture_model(self):
        data = {
            "step": self.step,
            "num_nodes": self.graph.get_graph_size(),
            "num_edges": self.graph.get_graph_edge(),
            "mse": self.total_mse / self.step,
            "success_rate": self.true_detect / self.step
        }
        self.graph.capture_data.append(data)

        if (self.step % 2000 == 0):
            self.export_log()



    def fit(self, epoch=1):
        self.export_parameter()
        dataset_path = DATASET[self.dataset_name]
        dataset_path = "./Disease_dataset/Raw/Dataset3_Official/NumpyData_3031/train/"

        t_ins = 0
        for i in range(epoch):
            for filename in os.listdir(dataset_path):
                if (filename.endswith(".npy")):
                    file_path = os.path.join(dataset_path, filename)
                    ecg_array = np.load(file_path)

                    parts = filename.split('_')
                    if len(parts) >= 3:
                        timestamp, label, file_type = parts

                    label = int(label)
                    
                    output = np.zeros((1, 4), dtype=int)
                    if 0 <= label < 4:
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

                    if (np.all(softmax(input_node.actual_output) == input_node.target)):
                        self.true_detect += 1

                    t_ins += 1
                    
                    if (t_ins == MODEL_CONSTANT.THETA * self.graph.get_graph_size()):
                        print("STEP", self.step)
                        self.graph.log[self.step] = "insert"
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

                    if (self.step % MODEL_CONSTANT.CAPTURE_TIME == 0):
                        self.capture_model()

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



dataset_name = "0410"
dataset_log_path = "./Dataset_log/" + dataset_name + "/"
model = LLCS(dataset_name=dataset_name, dataset_log_path=dataset_log_path)
model.fit(epoch = 2)

print(model.graph.log)
