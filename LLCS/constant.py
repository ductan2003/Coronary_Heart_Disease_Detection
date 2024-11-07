class MODEL_CONSTANT:
    # INPUT_ADAPTATION_THRESHOLD = 0.001

    # LEARNING_RATE_FOR_BEST_NODE = 0.1
    # LEARNING_RATE_FOR_NEIGHBOR = 0.01

    # OUTPUT_ADAPTATION_THRESHOLD= -0.05
    # OUTPUT_ADAPTATION_LEARNING_RATE= 0.15

    # INSERTION_TOLERANCE = 0.05
    # INSERTION_LEARNING_RATE = 0.2

    # INPUT_ADAPTATION_THRESHOLD = 0.05
    # LEARNING_RATE_FOR_BEST_NODE = 0.1
    # LEARNING_RATE_FOR_NEIGHBOR = 0.01
    # OUTPUT_ADAPTATION_THRESHOLD = -0.05
    # OUTPUT_ADAPTATION_LEARNING_RATE = 0.15
    # INSERTION_TOLERANCE = 0.05
    # INSERTION_LEARNING_RATE = 0.1

    DELETION_THRESHOLD = 0.05
    MINIMAL_AGE = 5
    SUFFICIENT_STABILIZATION= 0.95

    INPUT_ADAPTATION_THRESHOLD = 0.02
    LEARNING_RATE_FOR_BEST_NODE = 0.2
    LEARNING_RATE_FOR_NEIGHBOR = 0.02
    OUTPUT_ADAPTATION_THRESHOLD = -0.03
    OUTPUT_ADAPTATION_LEARNING_RATE = 0.2
    INSERTION_TOLERANCE = 0.01
    INSERTION_LEARNING_RATE = 0.1

    TL = 15
    TS = 5
    TY = 50
    TV = 50

    MAXIMUM_EDGE_AGE = 150

    THETA = 15

    CAPTURE_TIME = 2

DATASET = {
    "Overlap_testing": "./Disease_dataset/Overlap_testing/NumpyData/train/",
    "Overlap_testing_5000": "./Disease_dataset/Overlap_testing_5000/NumpyData/",
    "NonOverlap_testing": "./Disease_dataset/NonOverlap_testing/NumpyData/train/",
    "Ex1_NonOverlap_3000": "./Disease_dataset/Ex1_NonOverlap_3000/NumpyData/",
    "Ex1_Overlap_3000": "./Disease_dataset/Ex1_Overlap_3000/NumpyData/"
}
    