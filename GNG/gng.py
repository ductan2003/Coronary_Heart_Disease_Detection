import os
import numpy as np

class GrowingNeuralGas:
    def __init__(self, input_dim, max_neurons=100, max_age=50, epsilon_b=0.05, epsilon_n=0.006, alpha=0.5, beta=0.0005, lambda_=100):
        """
        Initializes a Growing Neural Gas (GNG) model.
        
        Parameters:
        - input_dim: Dimension of the input data.
        - max_neurons: Maximum number of neurons to add.
        - max_age: Maximum age of connections.
        - epsilon_b: Learning rate for the winning neuron.
        - epsilon_n: Learning rate for neighbors of the winning neuron.
        - alpha: Error reduction factor for the neurons.
        - beta: Error decay factor for all neurons after each iteration.
        - lambda_: Number of iterations between neuron insertions.
        """
        self.input_dim = input_dim
        self.max_neurons = max_neurons
        self.max_age = max_age
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        
        # Initialize neurons and other properties
        self.neurons = [np.random.rand(input_dim) for _ in range(2)]
        self.errors = np.zeros(len(self.neurons))
        self.edges = {}  # Connection age between neurons
        self.iteration = 0

    def _find_nearest_neurons(self, x):
        """
        Find the indices of the two closest neurons to the input vector x.
        """
        distances = np.linalg.norm(np.array(self.neurons) - x, axis=1)
        return np.argsort(distances)[:2]  # Return indices of two closest neurons

    def _add_neuron(self):
        """
        Add a new neuron between the neuron with the highest error and its farthest neighbor.
        """
        q = np.argmax(self.errors)
        f = max(self.edges[q], key=lambda n: np.linalg.norm(self.neurons[q] - self.neurons[n]))
        new_neuron = (self.neurons[q] + self.neurons[f]) / 2
        
        # Update structures with the new neuron
        self.neurons.append(new_neuron)
        self.errors = np.append(self.errors, self.errors[q] * self.alpha)
        self.errors[q] *= self.alpha
        self.errors[f] *= self.alpha
        self.edges[len(self.neurons) - 1] = {}
        self.edges[q][len(self.neurons) - 1] = 0
        self.edges[f][len(self.neurons) - 1] = 0

    def _update_errors(self):
        """
        Reduce the error of all neurons by beta.
        """
        self.errors *= (1 - self.beta)

    def _prune_edges(self):
        """
        Remove connections that exceed max_age.
        """
        for i, neighbors in list(self.edges.items()):
            for j, age in list(neighbors.items()):
                if age > self.max_age:
                    del self.edges[i][j]
                    if j in self.edges and i in self.edges[j]:
                        del self.edges[j][i]
            if not self.edges[i]:
                del self.edges[i]  # Remove neuron if no neighbors left

    def train(self, X, epochs=1):
        """
        Train the Growing Neural Gas network on the dataset X.
        
        Parameters:
        - X: Input data, a 2D array where each row is a sample.
        - epochs: Number of epochs to train for.
        """
        for epoch in range(epochs):
            for x in X:
                # Step 1: Find the two nearest neurons
                s1, s2 = self._find_nearest_neurons(x)

                # Step 2: Update the winning neuron and its neighbors
                self.neurons[s1] += self.epsilon_b * (x - self.neurons[s1])
                for neighbor in self.edges.get(s1, {}):
                    self.neurons[neighbor] += self.epsilon_n * (x - self.neurons[neighbor])
                    self.edges[s1][neighbor] += 1

                # Step 3: Update the error for the winning neuron
                self.errors[s1] += np.linalg.norm(x - self.neurons[s1])

                # Step 4: Reset the age of the edge between s1 and s2
                self.edges.setdefault(s1, {})[s2] = 0
                self.edges.setdefault(s2, {})[s1] = 0

                # Step 5: Add new neurons every lambda_ iterations
                if self.iteration % self.lambda_ == 0 and len(self.neurons) < self.max_neurons:
                    self._add_neuron()

                # Step 6: Decrease errors and prune old edges
                self._update_errors()
                self._prune_edges()

                # Increment iteration count
                self.iteration += 1

        print(f"Training complete after {epochs} epochs")

    def train_from_folder(self, folder_path, epochs=1):
        """
        Train the GNG model on data from .npy files in a specified folder.

        Parameters:
        - folder_path: Path to the folder containing .npy files.
        - epochs: Number of epochs to train for.
        """
        for filename in os.listdir(folder_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(folder_path, filename)
                data = np.load(file_path)
                print(f"Training on {filename} with {data.shape[0]} samples.")
                self.train(data, epochs)

    def evaluate(self, val_folder_path):
        """
        Evaluate the GNG model on validation data from a specified folder.
        
        Parameters:
        - val_folder_path: Path to the folder containing validation .npy files.

        Returns:
        - avg_error: Average reconstruction error over the validation dataset.
        """
        total_error = 0
        sample_count = 0
        for filename in os.listdir(val_folder_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(val_folder_path, filename)
                data = np.load(file_path)
                sample_count += len(data)
                
                # Calculate error for each sample in the validation set
                for x in data:
                    s1 = self._find_nearest_neurons(x)[0]  # Closest neuron
                    total_error += np.linalg.norm(x - self.neurons[s1])
        
        avg_error = total_error / sample_count if sample_count > 0 else 0
        print(f"Average reconstruction error on validation set: {avg_error:.4f}")
        return avg_error

# Example usage:
# Initialize GNG with input dimension based on the data
# input_dim = 30  # Assuming each data sample is a 30-dimensional vector, as in the example file
# gng = GrowingNeuralGas(input_dim=input_dim)

# # Train the model using the data in the specified folder
# train_folder_path = "../LLCS/Dataset1_Official/NumpyData/train/"  # Update this path
# gng.train_from_folder(train_folder_path, epochs=5)

# # Evaluate the model on the validation dataset
# val_folder_path = "../LLCS/Dataset1_Official/NumpyData/eval/"  # Update this path
# gng.evaluate(val_folder_path)
