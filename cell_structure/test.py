import numpy as np

# initializing points in
# numpy arrays
point1 = np.array([[0.2, 0.4, 0.3], [0.5, 0.3, 0.6], [0.3, 0.1, 0.2]])
point2 = np.array([[0.923, 0.923, 0.726]])

# calculating Euclidean distance
# using linalg.norm()
# dist = np.linalg.norm(point1 - point2)

# printing Euclidean distance
print(point1.dot(point2.T))