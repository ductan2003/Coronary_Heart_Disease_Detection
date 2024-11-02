import numpy as np
import os

class GrowingNeuralGas:
    # def __init__(self, input_dim, num_classes, max_neurons=100, max_age=50, epsilon_b=0.05, epsilon_n=0.006, alpha=0.5, beta=0.0005, lambda_=100):
    #     """
    #     Initializes a Growing Neural Gas (GNG) model with a fixed number of classes.
        
    #     Parameters:
    #     - input_dim: Dimension of the input data.
    #     - num_classes: Fixed number of classes for classification.
    #     - max_neurons: Maximum number of neurons to add.
    #     - max_age: Maximum age of connections.
    #     - epsilon_b: Learning rate for the winning neuron.
    #     - epsilon_n: Learning rate for neighbors of the winning neuron.
    #     - alpha: Error reduction factor for the neurons.
    #     - beta: Error decay factor for all neurons after each iteration.
    #     - lambda_: Number of iterations between neuron insertions.
    #     """
    #     self.input_dim = input_dim
    #     self.num_classes = num_classes
    #     self.max_neurons = max_neurons
    #     self.max_age = max_age
    #     self.epsilon_b = epsilon_b
    #     self.epsilon_n = epsilon_n
    #     self.alpha = alpha
    #     self.beta = beta
    #     self.lambda_ = lambda_
        
    #     # Initialize neurons and other properties
    #     self.neurons = [np.random.rand(input_dim) for _ in range(2)]
    #     print(len(self.neurons))
    #     self.errors = np.zeros(len(self.neurons))
    #     self.edges = {}  # Connection age between neurons
    #     self.iteration = 0
    #     self.neuron_labels = np.full(len(self.neurons), -1)  # Label each neuron (-1 for uninitialized)
    #     self.step_log = []  # List to capture the log of nodes and edges

    def __init__(self, input_dim=900, num_classes=2, max_neurons=100, max_age=50, epsilon_b=0.05, epsilon_n=0.006, alpha=0.5, beta=0.0005, lambda_=100):
        """
        Initializes a Growing Neural Gas (GNG) model with a fixed number of classes.
        
        Parameters:
        - input_dim: Dimension of the input data, default set to 900 for flattened 30x30 matrices.
        """
        self.input_dim = input_dim  # Should be 900 if input data is 30x30 flattened
        self.num_classes = num_classes
        self.max_neurons = max_neurons
        self.max_age = max_age
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        
        # Initialize neurons with the correct input dimension
        self.neurons = [np.random.rand(input_dim) for _ in range(2)]  # Each neuron is a 900-dimensional vector
        self.errors = np.zeros(len(self.neurons))
        self.edges = {}
        self.iteration = 0
        self.neuron_labels = np.full(len(self.neurons), -1)  # Label each neuron (-1 for uninitialized)
        self.step_log = []

    def get_log(self):
        return self.step_log

    def _log_step(self, step):
        """
        Log the number of nodes and edges at the given step.
        """
        num_nodes = len(self.neurons)
        num_edges = sum(len(neighbors) for neighbors in self.edges.values()) // 2  # Each edge is bidirectional
        self.step_log.append({"index": step, "num_nodes": num_nodes, "edge": num_edges})

    def _find_nearest_neurons(self, x):
        distances = np.linalg.norm(np.array(self.neurons) - x, axis=1)
        return np.argsort(distances)[:2]  # Return indices of two closest neurons

    # def _add_neuron(self):
    #     q = np.argmax(self.errors)
    #     f = max(self.edges[q], key=lambda n: np.linalg.norm(self.neurons[q] - self.neurons[n]))
    #     new_neuron = (self.neurons[q] + self.neurons[f]) / 2
        
    #     # Update structures with the new neuron
    #     self.neurons.append(new_neuron)
    #     self.errors = np.append(self.errors, self.errors[q] * self.alpha)
    #     self.errors[q] *= self.alpha
    #     self.errors[f] *= self.alpha
    #     self.edges[len(self.neurons) - 1] = {}
    #     self.edges[q][len(self.neurons) - 1] = 0
    #     self.edges[f][len(self.neurons) - 1] = 0
    #     self.neuron_labels = np.append(self.neuron_labels, -1)  # Initialize label as -1

    def _add_neuron(self):
        q = np.argmax(self.errors)
        
        # Check if the neuron with the highest error has neighbors
        if q not in self.edges or not self.edges[q]:
            print(f"Neuron {q} has no neighbors. Skipping addition of new neuron.")
            return  # Skip adding a new neuron if no neighbors are available
        
        # Find the neighbor of neuron q that is farthest away
        f = max(self.edges[q], key=lambda n: np.linalg.norm(self.neurons[q] - self.neurons[n]))
        new_neuron = (self.neurons[q] + self.neurons[f]) / 2
        
        # Add the new neuron and initialize its edges
        self.neurons.append(new_neuron)
        self.errors = np.append(self.errors, self.errors[q] * self.alpha)
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        new_neuron_index = len(self.neurons) - 1
        self.edges[new_neuron_index] = {}
        
        # Ensure both neurons q and f have edges dictionaries initialized
        if q not in self.edges:
            self.edges[q] = {}
        if f not in self.edges:
            self.edges[f] = {}
        
        # Create connections for the new neuron
        self.edges[q][new_neuron_index] = 0
        self.edges[f][new_neuron_index] = 0
        self.edges[new_neuron_index][q] = 0
        self.edges[new_neuron_index][f] = 0
        self.neuron_labels = np.append(self.neuron_labels, -1)



    def _update_errors(self):
        self.errors *= (1 - self.beta)

    def _prune_edges(self):
        for i, neighbors in list(self.edges.items()):
            for j, age in list(neighbors.items()):
                if age > self.max_age:
                    del self.edges[i][j]
                    if j in self.edges and i in self.edges[j]:
                        del self.edges[j][i]
            if not self.edges[i]:
                del self.edges[i]  # Remove neuron if no neighbors left

    def _extract_label_from_filename(self, filename):
        """
        Extract the label from the filename. Assumes filename format: timestamp_label_activitytype.npy.
        """
        try:
            return int(filename.split("_")[1])  # Extracts the label as an integer
        except (IndexError, ValueError):
            return -1  # Return -1 if label extraction fails

    # def train_from_folder(self, folder_path, epochs=1, log_interval=5):
    #     """
    #     Train the GNG model on data from .npy files in a specified folder.
    #     Each file is assumed to represent one of the fixed classes, with filename format: timestamp_label_activitytype.npy.
        
    #     Parameters:
    #     - folder_path: Path to the folder containing .npy files.
    #     - epochs: Number of epochs to train for.
    #     """
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith(".npy"):
    #             file_path = os.path.join(folder_path, filename)
    #             label = self._extract_label_from_filename(filename)
    #             if label == -1 or label >= self.num_classes:
    #                 print(f"Skipping file due to invalid label: {filename}")
    #                 continue

    #             data = np.load(file_path)
    #             print(f"Training on {filename} with {data.shape[0]} samples (label={label}).")
    #             for epoch in range(epochs):
    #                 for x in data:
    #                     # print("here more", x)
    #                     # print(self.neurons)
    #                     # Step 1: Find the two nearest neurons
    #                     s1, s2 = self._find_nearest_neurons(x)
                        

    #                     # Update neuron label for the winning neuron
    #                     self.neuron_labels[s1] = label

    #                     # Step 2: Update the winning neuron and its neighbors
    #                     self.neurons[s1] += self.epsilon_b * (x - self.neurons[s1])
    #                     for neighbor in self.edges.get(s1, {}):
    #                         self.neurons[neighbor] += self.epsilon_n * (x - self.neurons[neighbor])
    #                         self.edges[s1][neighbor] += 1

    #                     # Step 3: Update the error for the winning neuron
    #                     self.errors[s1] += np.linalg.norm(x - self.neurons[s1])

    #                     # Step 4: Reset the age of the edge between s1 and s2
    #                     self.edges.setdefault(s1, {})[s2] = 0
    #                     self.edges.setdefault(s2, {})[s1] = 0

    #                     # Step 5: Add new neurons every lambda_ iterations
    #                     if self.iteration % self.lambda_ == 0 and len(self.neurons) < self.max_neurons:
    #                         self._add_neuron()

    #                     # Step 6: Decrease errors and prune old edges
    #                     self._update_errors()
    #                     self._prune_edges()

    #             if self.iteration % log_interval == 0:
    #                 self._log_step(self.iteration)


    #             # Increment iteration count
    #             self.iteration += 1

    def train_from_folder(self, folder_path, epochs=1, log_interval=5):
        """
        Train the GNG model on data from .npy files in a specified folder.
        Each file represents a single neuron input, with filename format: timestamp_label_activitytype.npy.
        
        Parameters:
        - folder_path: Path to the folder containing .npy files.
        - epochs: Number of epochs to train for.
        - log_interval: Interval at which to log the number of nodes and edges.
        """
        for epoch in range(epochs):
            for filename in os.listdir(folder_path):
                if filename.endswith(".npy"):
                    file_path = os.path.join(folder_path, filename)
                    
                    # Load the entire file as a single input and flatten it
                    data = np.load(file_path).flatten()  # Flatten to 1D if needed
                    label = self._extract_label_from_filename(filename)
                    
                    if label == -1 or label >= self.num_classes:
                        print(f"Skipping file due to invalid label: {filename}")
                        continue

                    print(f"Training on {filename} as a single input (label={label}).")
                    
                    # Step 1: Find the two nearest neurons for this input
                    s1, s2 = self._find_nearest_neurons(data)
                    
                    # Update neuron label for the winning neuron
                    self.neuron_labels[s1] = label

                    # Step 2: Update the winning neuron and its neighbors
                    self.neurons[s1] += self.epsilon_b * (data - self.neurons[s1])
                    for neighbor in self.edges.get(s1, {}):
                        self.neurons[neighbor] += self.epsilon_n * (data - self.neurons[neighbor])
                        self.edges[s1][neighbor] += 1

                    # Step 3: Update the error for the winning neuron
                    self.errors[s1] += np.linalg.norm(data - self.neurons[s1])

                    # Step 4: Reset the age of the edge between s1 and s2
                    self.edges.setdefault(s1, {})[s2] = 0
                    self.edges.setdefault(s2, {})[s1] = 0

                    # Step 5: Add new neurons every lambda_ iterations
                    if self.iteration % self.lambda_ == 0 and len(self.neurons) < self.max_neurons:
                        self._add_neuron()

                    # Step 6: Decrease errors and prune old edges
                    self._update_errors()
                    self._prune_edges()

                    # Log node and edge information every log_interval steps
                    if self.iteration % log_interval == 0:
                        self._log_step(self.iteration)

                    # Increment iteration count
                    self.iteration += 1

    # def evaluate(self, val_folder_path):
    #     """
    #     Evaluate the GNG model on validation data from a specified folder with fixed classes.
    #     Each .npy file in the folder represents one of the predefined classes with filename format: timestamp_label_activitytype.npy.
        
    #     Parameters:
    #     - val_folder_path: Path to the folder containing validation .npy files.

    #     Returns:
    #     - accuracy: Classification accuracy on the validation dataset.
    #     """
    #     correct_predictions = 0
    #     total_predictions = 0
    #     for filename in os.listdir(val_folder_path):
    #         if filename.endswith(".npy"):
    #             file_path = os.path.join(val_folder_path, filename)
    #             label = self._extract_label_from_filename(filename)
    #             if label == -1 or label >= self.num_classes:
    #                 print(f"Skipping file due to invalid label: {filename}")
    #                 continue

    #             data = np.load(file_path)

    #             # Classify each sample and count correct predictions
    #             for x in data:
    #                 s1 = self._find_nearest_neurons(x)[0]  # Closest neuron
    #                 predicted_label = self.neuron_labels[s1]
    #                 if predicted_label == label:
    #                     correct_predictions += 1
    #                 total_predictions += 1

    #     accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    #     print(f"Classification accuracy on validation set: {accuracy:.4f}")
    #     return accuracy

    def evaluate(self, val_folder_path):
        """
        Evaluate the GNG model on validation data from a specified folder with fixed classes.
        Each .npy file in the folder represents one of the predefined classes with filename format: timestamp_label_activitytype.npy.
        
        Parameters:
        - val_folder_path: Path to the folder containing validation .npy files.

        Returns:
        - accuracy: Classification accuracy on the validation dataset.
        """
        correct_predictions = 0
        total_predictions = 0
        
        for filename in os.listdir(val_folder_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(val_folder_path, filename)
                label = self._extract_label_from_filename(filename)
                if label == -1 or label >= self.num_classes:
                    print(f"Skipping file due to invalid label: {filename}")
                    continue

                # Load the entire file as a single input and flatten it
                data = np.load(file_path).flatten()
                
                # Find the nearest neuron to classify this input
                s1 = self._find_nearest_neurons(data)[0]  # Closest neuron
                predicted_label = self.neuron_labels[s1]
                
                # Check if the prediction is correct
                if predicted_label == label:
                    correct_predictions += 1
                total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Classification accuracy on validation set: {accuracy:.4f}")
        return accuracy

