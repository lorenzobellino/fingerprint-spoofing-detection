
from Utils.utils import load, plot
import numpy as np

def LDA(logger, m: int = 3) -> None:
    plotname = f"plots/LDA_v2_matrix_{m}.png"
    logger.info(
        "\n##########################################################\n#                                                        #\n#                    COMPUTING LDA v2                    #\n#                                                        #\n##########################################################"
    )
    d, c = load()
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
    logger.info(f"\n-------------------------Sb------------------------\n{Sb}")
    logger.info("Computing the covariance matrix within class (Sw)")
    Sw = sum(c * cov for c, cov in zip(nc, covariances)) / d.shape[1]
    logger.info(f"\n-------------------------Sw------------------------\n{Sw}")
    logger.info("Solving the generalized eigenvalue problem by joint diagonalization")
    eigvecs, eigval, _ = np.linalg.svd(Sw)

    P1 = np.dot(np.dot(eigvecs, np.diag(1.0 / (eigval**0.5))), eigvecs.T)

    logger.info("Computing the trasformed between class covariance (Sbt)")
    Sbt = P1 * Sb * P1.T
    logger.info(f"\n-------------------------Sbt------------------------\n{Sbt}")
    logger.info("Calculating the eigenvectors of Sbt")
    eigvecs, _, _ = np.linalg.svd(Sbt)
    logger.info(
        f"\n-------------------------eigvecs------------------------\n{eigvecs}"
    )
    logger.info(f"Retrieving m={m} largest eigenvectors ... ")
    # P2 = eigvecs[:, 0:m]
    P2 = eigvecs
    logger.info(f"\n--------------------------P-------------------------\n{P2}")
    logger.info("Calculating the LDA amtrix W")
    W = P1.T * P2
    logger.info(f"\n--------------------------W-------------------------\n{W}")
    proj = np.dot(W.T, d)
    logger.info("Plotting the data")
    plot(proj, c, plotname)



def main(args,logger):
    logger.info("Starting main function on LGA")
    LDA(logger,args.m)