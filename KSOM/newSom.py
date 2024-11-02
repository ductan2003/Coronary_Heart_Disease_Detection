import sys, math, time
import matplotlib.pyplot as plt
import numpy as np

class NewSOM:
    def __init__(self):
        # Initialize placeholders for map properties
        self.trained = False

    def create(self, width, height):
        self.x = width  # Map width
        self.y = height  # Map height
        self.trained = False

    def save(self, filename):
        # Save SOM to .npy file
        if self.trained:
            np.save(filename, self.node_vectors)
            return True
        else:
            return False

    def load(self, filename):
        # Load SOM from .npy file
        self.node_vectors = np.load(filename)
        self.x = self.node_vectors.shape[1]
        self.y = self.node_vectors.shape[0]
        self.ch = self.node_vectors.shape[2]
        self.trained = True
        return True

    def get_map_vectors(self):
        # Return the map vectors
        if self.trained:
            return self.node_vectors
        else:
            return False

    def flatten_input(self, input_data):
        # Flatten input to shape (n, ch), regardless of original shape
        return input_data.reshape(-1, self.ch)

    def distance(self, vect_a, vect_b):
        # Calculate Euclidean distance
        return np.linalg.norm(vect_a - vect_b)

    def initialize_map(self):
        # Initialize map weight vectors with a small random positive value
        ds_mul = np.mean(self.input_arr) / 0.5
        self.node_vectors = np.random.rand(self.y, self.x, self.ch) * ds_mul + 0.01

    def fit(self, input_arr, n_iter, batch_size=32, lr=0.1, random_sampling=1.0, 
            neighbor_dist=None, dist_method='euclidean'):
        # Automatically infer `ch` from input shape
        self.input_arr = input_arr
        self.ch = input_arr.shape[-1]  # Number of features in each vector
        input_arr = input_arr.reshape(-1, self.ch)

        self.n_iter = n_iter
        self.batch_size = batch_size
        self.dist_method = dist_method

        start_time = time.time()
        self.initialize_map()

        self.lr = lr
        self.lr_decay = 0.8  # lr decay per iteration
        if neighbor_dist is None:
            neighbor_dist = min(self.x, self.y) / 1.3
        self.nb_dist = int(neighbor_dist)
        self.nb_decay = 1.5

        # Pad the vector map for easy array processing
        tmp_node_vects = np.zeros((self.y + 2 * self.nb_dist, self.x + 2 * self.nb_dist, self.ch))
        tmp_node_vects[self.nb_dist : self.nb_dist + self.y, 
                       self.nb_dist : self.nb_dist + self.x] = self.node_vectors.copy()
        self.node_vectors = tmp_node_vects

        if random_sampling > 1 or random_sampling <= 0:
            random_sampling = 1
        n_data_pts = int(self.input_arr.shape[0] * random_sampling)
        data_idx_arr = np.arange(self.input_arr.shape[0])
        batch_count = math.ceil(n_data_pts / self.batch_size)

        for iteration in range(self.n_iter):
            self.make_neighbor_function(iteration)
            np.random.shuffle(data_idx_arr)
            total_dist, total_count, print_count = 0, 0, 0
            
            for batch in range(batch_count): 
                steps_left = n_data_pts - batch * self.batch_size
                steps_in_batch = self.batch_size if steps_left >= self.batch_size else steps_left
                
                bm_node_idx_arr = np.zeros((steps_in_batch, 3), dtype=np.int32)

                for step in range(steps_in_batch):
                    input_idx = data_idx_arr[batch * self.batch_size + step]
                    input_vect = self.input_arr[input_idx]
                    y, x, dist = self.find_best_matching_node(input_vect)
                    bm_node_idx_arr[step] = [y, x, input_idx]
                    total_dist += dist
                
                self.update_node_vectors(bm_node_idx_arr)
            print(f' Average distance = {total_dist / n_data_pts:0.5f}')
            self.lr *= self.lr_decay

        self.node_vectors = self.node_vectors[self.nb_dist : self.nb_dist + self.y, 
                                              self.nb_dist : self.nb_dist + self.x]
        del self.input_arr
        end_time = time.time()
        self.trained = True
        print(f'Training done in {end_time - start_time:0.6f} seconds.')

    def update_node_vectors(self, bm_node_idx_arr):
        size = self.nb_dist * 2 + 1
        for idx in range(bm_node_idx_arr.shape[0]):
            node_y = bm_node_idx_arr[idx, 0]
            node_x = bm_node_idx_arr[idx, 1]
            inp_idx = bm_node_idx_arr[idx, 2]
            input_vect = self.input_arr[inp_idx]

            # Extract a subarray around the best matching node
            subarray = self.node_vectors[node_y:node_y+size, node_x:node_x+size]
            
            # Expand input_vect to match the shape of subarray for broadcasting
            input_vect_expanded = np.tile(input_vect, (size, size, 1))

            # Apply the neighborhood weights and learning rate
            weight_adjusted = np.expand_dims(self.nb_weights, axis=-1) * self.lr * (input_vect_expanded - subarray)
            
            # Update the node vectors
            self.node_vectors[node_y:node_y+size, node_x:node_x+size] += weight_adjusted

    def find_best_matching_node(self, data_vect):
        min_dist, x, y = None, None, None
        for y_idx in range(self.y):
            for x_idx in range(self.x):
                node_vect = self.node_vectors[y_idx + self.nb_dist, x_idx + self.nb_dist]
                dist = self.distance(data_vect, node_vect)
                if min_dist is None or min_dist > dist:
                    min_dist, x, y = dist, x_idx, y_idx
        return y, x, min_dist

    def make_neighbor_function(self, iteration):
        size = self.nb_dist * 2 + 1
        sigma = size / (7 + iteration / self.nb_decay)
        self.nb_weights = np.zeros((size, size))
        cp = size // 2
        p1 = 1.0 / (2 * math.pi * sigma ** 2) 
        pdiv = 2.0 * sigma ** 2
        for y in range(size):
            for x in range(size):
                ep = -((x - cp) ** 2 + (y - cp) ** 2) / pdiv
                value = p1 * math.e ** ep
                self.nb_weights[y, x] = value

        # Normalize weights, avoiding division by zero
        max_weight = np.max(self.nb_weights)
        if max_weight > 0:
            self.nb_weights /= max_weight

# Initialize and create the SOM with the loaded dataset dimensions
# som = SOM()
# som.create(width=13, height=13)

# # Train the SOM with the loaded data
# som.fit(data, random_sampling=0.5, n_iter=20, dist_method='euclidean')
