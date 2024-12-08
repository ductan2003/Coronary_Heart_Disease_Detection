class MODEL_CONSTANT:
    DELETION_THRESHOLD = 0.02
    MINIMAL_AGE = 0.01
    SUFFICIENT_STABILIZATION= 0.01

    INPUT_ADAPTATION_THRESHOLD = 0.02
    LEARNING_RATE_FOR_BEST_NODE = 0.1
    LEARNING_RATE_FOR_NEIGHBOR = 0.003
    OUTPUT_ADAPTATION_THRESHOLD = -0.03
    OUTPUT_ADAPTATION_LEARNING_RATE = 0.15
    INSERTION_TOLERANCE = 0.005
    INSERTION_LEARNING_RATE = 0.2

    TL = 20
    TS = 5
    TY = 100
    TV = 100

    MAXIMUM_EDGE_AGE = 50
    MAXIMUM_NEW_NODE_AGE = 300

    THETA = 5

    CAPTURE_TIME = 100

DATASET = {
    "Overlap_testing": "./Disease_dataset/Overlap_testing/NumpyData/train/",
    "Overlap_testing_5000": "./Disease_dataset/Overlap_testing_5000/NumpyData/",
    "NonOverlap_testing": "./Disease_dataset/NonOverlap_testing/NumpyData/train/",
    "Ex1_NonOverlap_3000": "./Disease_dataset/Ex1_NonOverlap_3000/NumpyData/",
    "Ex1_Overlap_3000": "./Disease_dataset/Ex1_Overlap_3000/NumpyData/",
    "Ex2_Dataset3_2000": "./Disease_dataset/Ex2_Dataset3_2000/NumpyData/",
    "Ex2_Dataset4_1050": "./Disease_dataset/Ex2_Dataset4_1050/NumpyData/",
    "Ex2_Dataset5_450": "./Disease_dataset/Ex2_Dataset5_450/NumpyData/",
    "DatasetA": "/Users/tannguyen/Coronary_Heart_Disease_Detection/LLCS/Disease_dataset/OfficialDatasetA/NumpyData/train/",
    "DatasetB": "/Users/tannguyen/Coronary_Heart_Disease_Detection/LLCS/Disease_dataset/OfficialDatasetB/NumpyData/train/",
    "Env1": "/Users/tannguyen/Coronary_Heart_Disease_Detection/LLCS/Disease_dataset/Env1/NumpyData/",
    "Env2": "/Users/tannguyen/Coronary_Heart_Disease_Detection/LLCS/Disease_dataset/Env2/NumpyData/",
    "Env3": "/Users/tannguyen/Coronary_Heart_Disease_Detection/LLCS/Disease_dataset/Env3/NumpyData/",
    "Env4": "/Users/tannguyen/Coronary_Heart_Disease_Detection/LLCS/Disease_dataset/Env4/NumpyData/",
    "RandomDataset": "/Users/tannguyen/Coronary_Heart_Disease_Detection/LLCS/Disease_dataset/RandomDataset/NumpyData/",
}
    