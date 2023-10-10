
from Utils.utils import load, plot, split_db_2to1
import numpy as np

def PCA(logger, D, m: int):
    logger.debug("Calculating the mean and centering data ... ")
    mean = np.mean(D, axis=1)
    logger.debug(f"\n-------------------------mean-----------------------\n{mean}")
    centerd_data = D - mean.reshape((mean.size, 1))
    logger.debug("Done centering")
    logger.debug("Computing covariance matrix ... ")
    cov = np.dot(centerd_data, centerd_data.T) / D.shape[1]
    logger.debug(f"\n-------------------------cov------------------------\n{cov}")
    logger.debug("Computing eigenvectors ... ")
    eigvecs, eigval, _ = np.linalg.svd(cov)
    logger.debug("Done computing eigenvectors")
    logger.debug(f"Retrieving m={m} largest eigenvectors ... ")
    P = eigvecs[:, 0:m]
    logger.debug(f"\n--------------------------P-------------------------\n{P}")
    return P,eigval


def main(args,logger):
    logger.info("Starting main function on PCA")
    D,L = load(logger)
    # DTR,DTE,LTR,LTE = split_db_2to1(D,L)
    logger.info("Done splitting the dataset")
    # d0 = DTR[:, LTR == 0]
    # d1 = DTR[:, LTR == 1]
    
    P,_ = PCA(logger, D, args.m)
    logger.info(f"Done retrieving m={args.m} largest eigenvectors")
    logger.info(f"Projecting data onto m={args.m} eigenvectors ... ")
    proj = np.dot(P.T, D)
    logger.info(f"Done projecting data onto m={args.m} eigenvectors")
    logger.info("Plotting the data ... ")
    plot(proj, L, f"plots/PCA_matrix_{args.m}.png")
    logger.info("Done")


