import numpy as np

def load():
    print("loading ...")
    dataset = np.loadtxt("data/Train.txt", delimiter=",", dtype=np.float32, usecols= range(10))
    classes = np.loadtxt("data/Train.txt", delimiter=",", dtype=np.int32, usecols=10)
    return dataset, classes