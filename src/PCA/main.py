
from Utils.utils import load, plot
import numpy as np

def PCA(logger, m: int) -> None:
    logger.info(
        "\n##########################################################\n#                                                        #\n#                    COMPUTING PCA                       #\n#                                                        #\n##########################################################"
    )
    plotname = f"plots/PCA_v2_matrix_{m}.png"
    d, c = load(logger)
    logger.info("Done loading")
    logger.info("Calculating the mean and centering data ... ")
    mean = np.mean(d, axis=1)
    logger.debug(f"\n-------------------------mean-----------------------\n{mean}")
    centerd_data = d - mean.reshape((mean.size, 1))
    logger.info("Done centering")
    logger.info("Computing covariance matrix ... ")
    cov = np.dot(centerd_data, centerd_data.T) / d.shape[1]
    logger.debug(f"\n-------------------------cov------------------------\n{cov}")
    logger.info("Computing eigenvectors ... ")
    eigvecs, _, _ = np.linalg.svd(cov)
    logger.info("Done computing eigenvectors")
    logger.info(f"Retrieving m={m} largest eigenvectors ... ")
    P = eigvecs[:, 0:m]
    logger.debug(f"\n--------------------------P-------------------------\n{P}")
    logger.info(f"Done retrieving m={m} largest eigenvectors")
    logger.info(f"Projecting data onto m={m} eigenvectors ... ")
    proj = np.dot(P.T, d)
    logger.info(f"Done projecting data onto m={m} eigenvectors")
    logger.info("Plotting the data ... ")
    plot(proj, c, plotname)


def main(args,logger):
    logger.info("Starting main function on PCA")
    PCA(logger, args.m)

