
from Utils.utils import load, plot, plot_hist
import numpy as np
from scipy import linalg

def LDA(logger, m: int) -> None:
    plotname = f"plots/LDA_v2_matrix_{m}.png"
    logger.info(
        "\n##########################################################\n#                                                        #\n#                    COMPUTING LDA                        #\n#                                                        #\n##########################################################"
    )
    d, c = load(logger)
    logger.info("Separating the dataset in each class")
    d0 = d[:, c == 0]
    d1 = d[:, c == 1]
    nc = [d0.shape[1], d1.shape[1]]
    logger.info("Computing the covariance matrix for each class")
    mean = np.mean(d, axis=1)
    mean0 = np.mean(d0, axis=1)
    mean1 = np.mean(d1, axis=1)
    class_means = [mean0, mean1]
    centerd_data0 = d0 - mean0.reshape((mean0.size, 1))
    centerd_data1 = d1 - mean1.reshape((mean1.size, 1))
    cov0 = np.dot(centerd_data0, centerd_data0.T) / d0.shape[1]
    cov1 = np.dot(centerd_data1, centerd_data1.T) / d1.shape[1]
    covariances = [cov0, cov1]
    logger.info("Computing the covariance matrix between class (Sb)")
    Sb = (
        sum(
            c * (m - mean) * (m - mean).reshape((m.size, 1))
            for c, m in zip(nc, class_means)
        )
        / d.shape[1]
    )
    logger.debug(f"\n-------------------------Sb------------------------\n{Sb}")
    logger.info("Computing the covariance matrix within class (Sw)")
    Sw = sum(c * cov for c, cov in zip(nc, covariances)) / d.shape[1]
    
    logger.info(f"\n-------------------------Sw------------------------\n{Sw}")
    logger.info("Solving the generalized eigenvalue problem by joint diagonalization")
    eigvecs, eigval, _ = np.linalg.svd(Sw)
    logger.info(f"\n-------------------------Sw------------------------\n{Sw}")
    logger.info("Solving the generalized eigenvalue problem")
    # cannot use np.linalg.eigh
    # because it does not support generalized eigenvalue problem
    eigvals, eigvecs = linalg.eigh(Sb, Sw)
    U = eigvecs[:, ::-1]
    logger.info(f"Retrieving {m} largest eigenvectors")
    W = U[:, 0:m]
    logger.info(f"\n--------------------------W-------------------------\n{W}")
    logger.info(f"Projecting data onto {m} eigenvectors")
    proj = np.dot(W.T, d)
    logger.info("Plotting the data")
    plot_hist(proj, c, f"plots/LDA_v2_hist_{m}.png")

    



def main(args,logger):
    logger.info("Starting main function on LGA")
    LDA(logger,args.m)