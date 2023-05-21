import numpy as np
import matplotlib.pyplot as plt
from Utils.utils import vrow

def loglikelihood(X, mu, C) -> np.array:
    """Compute the log-likelihood of a set of samples

    Args:
        X (np.Array): (M,N) matrix of N samples of dimension M
        mu (np.Array): array of shape (M,1) representing the mean
        C (np.Array): array of shape (M,M) representig the covariance matrix
    """
    return logpdf_GAU_ND_fast(X, mu, C).sum()

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


def MLE(logger) -> None:
    """Compute the Maximum Likelihood Estimate"""
    logger.info("Calculating the Maximum Likelihood Estimate ...")
    logger.info("Loading samples ...")
    XND = np.load("solutions/XND.npy")
    logger.info("computing mean and covariance ...")
    mu = XND.mean(axis=1).reshape(-1, 1)
    XNDC = XND - mu  # centering the data
    C = np.dot(XNDC, XNDC.T) / XND.shape[1]
    logger.info(f"mu = \n{mu}")
    logger.info(f"C = \n{C}")
    logger.info("Computing the log-likelihood ...")
    ll = loglikelihood(XND, mu, C)
    logger.info(f"Log-likelihood = {ll}")
    logger.info("Loading an example Dataset ...")
    # X1D = np.load("solutions/X1D.npy")
    logger.info("computing mean and covariance ...")
    mu = X1D.mean(axis=1).reshape(-1, 1)
    X1DC = X1D - mu  # centering the data
    C = np.dot(X1DC, X1DC.T) / X1D.shape[1]
    logger.info(f"mu = \n{mu}")
    logger.info(f"C = \n{C}")
    logger.info("Computing the log-likelihood ...")
    ll = loglikelihood(X1D, mu, C)
    logger.info(f"Log-likelihood = {ll}")
    logger.info("Plotting the solution ...")
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_fast(vrow(XPlot), mu, C)))
    plt.savefig("./plots/ll1D.png")

def main(args,logger)->None:
    MLE(logger)