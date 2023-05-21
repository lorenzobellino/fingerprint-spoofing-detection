import numpy as np
import matplotlib.pyplot as plt

def load(logger):
    logger.info("Loading the dataset")
    d = []
    c = []
    with open("data/Train.txt", "r") as f:
        for line in f:
            d.append(
                np.array([float(_) for _ in line.strip().split(",")[:-1]]).reshape(10, 1)
            )
            c.append(int(line.strip().split(",")[-1]))
    # dataset = np.loadtxt("data/Train.txt", delimiter=",", dtype=np.float32, usecols= range(10))
    # classes = np.loadtxt("data/Train.txt", delimiter=",", dtype=np.int32, usecols=10)
    dataset = np.hstack(d)
    classes = np.array(c)
    return dataset, classes


def plot(proj: np.array, c: np.array, filename: str) -> None:
    plt.figure()
    plt.scatter(proj[0, c == 0], proj[1, c == 0], label="Original")
    plt.scatter(proj[0, c == 1], proj[1, c == 1], label="Spoofed")
    plt.legend()
    # plt.show()
    plt.savefig(filename)