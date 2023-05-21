import numpy as np
from Utils.utils import vrow
import matplotlib.pyplot as plt

def logpdf_GAU_ND_fast(X, mu, C) -> np.array:
    """Generate the multivariate Gaussian Density for a Matrix of N samples

    Args:
        X (np.Array): (M,N) matrix of N samples of dimension M
        mu (np.Array): array of shape (M,1) representing the mean
        C (np.Array): array of shape (M,M) representig the covariance matrix
    """
    XC = X - mu
    M = X.shape[0]
    invC = np.linalg.inv(C)
    _, logDetC = np.linalg.slogdet(C)
    v = (XC * np.dot(invC, XC)).sum(0)

    lpdf = -(M / 2) * np.log(2 * np.pi) - (1 / 2) * logDetC - (1 / 2) * v
    return lpdf

def MGD(logger):
    raise NotImplementedError
    """Compute the multivariate Gaussian Density"""
    logger.info("Calculating the multivariate Gaussian density and plotting ...")
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1, 1)) * 1.0
    C = np.ones((1, 1)) * 2.0
    logger.debug(f"m = {m}")
    logger.debug(f"C = {C}")
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_fast(vrow(XPlot), m, C)))
    plt.savefig("./plots/llGAU.png")
    # logger.info("Comparing solutions ...")
    # pdfSol = np.load("solutions/llGAU.npy")
    # pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
    # logger.info(f"difference in solution1 = {np.abs(pdfSol - pdfGau).max()}")
    # XND = np.load("solutions/XND.npy")
    # mu = np.load("solutions/muND.npy")
    # C = np.load("solutions/CND.npy")
    # pdfSol = np.load("solutions/llND.npy")
    # pdfGau = logpdf_GAU_ND_fast(XND, mu, C)
    # logger.info(f"difference in solution2 = {np.abs(pdfSol - pdfGau).max()}")


def main(args,logger) -> None:
    MGD(logger)