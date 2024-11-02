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

    def fit(self):
        # df = pd.read_csv("./New_Training_Dataset__5000_samples_.csv")
        # df = pd.read_csv("./disease.csv")
        datasetPath = "./Dataset3_Official/NumpyData/train/"

        t_ins = 0
        index = 0
        # for index, row in df.iterrows():
        for filename in os.listdir(datasetPath):
            print("Index", index)
            if (filename.endswith(".npy")):
                file_path = os.path.join(datasetPath, filename)
                ecg_array = np.load(file_path)

                parts = filename.split('_')
                if len(parts) >= 3:
                    timestamp, label, file_type = parts

                label = int(label)
                
                # rgb_values = row[["Age","Fever","Cough","Fatigue","Shortness_of_Breath","Test_Result"]].to_numpy().reshape(1, 6)
                # label = row['Disease_Detected']
                
                # Create the output array based on the label
                if label == 0:
                    output = np.array([[1, 0]])
                else:
                    output = np.array([[0, 1]])
            
                input_node = Node(ecg_array, target=output)

                # 1.1. find best and second match
                first, second = model.graph.find_best_and_second_best_node(input_node)
                # print("DISPLAY FIRST AND SECOND")
                # first.display()
                # second.display()

                # update quality measure for learning
                first.update_input_weight(input_node, best_node=True)
                for neighbor in first.get_neighbors():
                    neighbor.get_node().update_input_weight(input_node, best_node=False)

                self.graph.activate_node(input_node)
                self.graph.update_out_weight(input_node)

                # print(self.graph.get_graph_size())
                # if (index == self.graph.get)
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
                    }
                    self.graph.capture_data.append(data)
            # if (index == 500):
            #     break
        # self.graph.display()
        
    def evaluate(self):
        count = 0
        # df = pd.read_csv("./New_Test_Dataset__1000_samples_.csv")
        datasetPath = "./Dataset3_Official/NumpyData/val/"

        index = 0
        # for index, row in df.iterrows():
        for filename in os.listdir(datasetPath):
            if (filename.endswith(".npy")):
                index += 1
                print("Index", index)
                file_path = os.path.join(datasetPath, filename)
                ecg_array = np.load(file_path)

                parts = filename.split('_')
                if len(parts) >= 3:
                    timestamp, label, file_type = parts

                print("label", label)
                label = int(label)

            # df = pd.read_csv("./Test_Dataset.csv")
            # for index, row in df.iterrows():
                # rgb_values = row[["Age","Fever","Cough","Fatigue","Shortness_of_Breath","Test_Result"]].to_numpy().reshape(1, 6)
                # label = row['Disease_Detected']

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
                
                print(detect)

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

# print(model.graph.capture_data)
# print(model.graph.get_graph_size())
# for item in model.graph.graph:
#     print(item.display_neighbor())

# a = Node(input_weight=np.array([[ 255,42, 63]]))

# first, second = model.graph.find_best_and_second_best_node(a)

# print("winner output ", first.display())
# model.graph.activate_node(a)

# output = model.graph.get_actual_output()

# print(output)

# print(np.array([[1, 2], [3, 4], [5, 6]]))

# a = Node(np.array([[ 12,72, 244]]))
# b = Node(np.array([ [11,32, 284]]))
# c = Node(np.array([ [12,22, 274]]))
# d = Node(np.array([ [52,78, 144]]))

# a.add_neighbor(b)
# a.add_neighbor(c)

# b.delete_neighbor_by_node(a)

# print("eghe")
# a.display_neighbor()
# b.display_neighbor()


# b.delete_neighbor()
# b = Node(np.array([[1, 3], [8, 2], [8, 1]]))
# c = Node(np.array([[2, 1], [3, 7], [2, 3]]))
# d = Node(np.array([[9, 5], [9, 3], [3, 6]]))

# model.graph.add_node(a)
# a.display()
# model.graph.add_node(b)
# b.display()
# model.graph.add_node(c)
# c.display()
# model.graph.add_node(d)
# d.display()
# print("_______________")
# a.add_neighbor(b)
# d.add_neighbor(a)

# model.graph.display()

# input_node = Node(np.array([[2, 4, 9]]))
# first, second = model.graph.find_best_and_second_best_node(input_node)

# first.update_input_weight(input_node, best_node = True)
# second.update_input_weight(input_node)

# model.graph.activate_node(input_node)

# print(model.graph.get_actual_output())

# model.graph.update_out_weight()

# model.graph.update_BI()
# model.graph.get_q_and_f_node()

# model.graph.check_deletion_criteria()
# first.actual_output = model.graph.get_actual_output()
# first.update_error_counter()
