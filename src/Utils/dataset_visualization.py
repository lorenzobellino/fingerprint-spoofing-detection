
# from Utils.utils import load, plot_correlations, randomize, plot_features_histograms,plot_explained_variance
from Utils.utils import  *
from PCA.main import PCA
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA as pcaa

def visualizeDataset(logger):
    logger.info(
        "\n##########################################################\n#                                                        #\n#                    VISUALIZATION                        #\n#                                                        #\n##########################################################"
    )
    D,C = load(logger)
    DTR , LTR = randomize(D, C)
    logger.info("Plotting Features Histograms")
    plot_features_histograms(DTR, LTR, "features_histogram")
    logger.info("Plotting Correlation")
    plot_correlations(DTR, "correlation")
    logger.info("Plotting Correlation Spoofed Fingerprint")
    plot_correlations(DTR[:, LTR == 0], "correlation_spoofedFingerprint", cmap="Reds")
    logger.info("Plotting Correlation Fingerprint")
    plot_correlations(DTR[:, LTR == 1], "correlation_fingerprint", cmap="Greens")
    logger.info("Plotting Explained Variance")
    plot_explained_variance(logger,DTR,PCA)
    logger.info("Done")
    # plot_features_scatterplots(DTR, LTR)


def main(args,logger )-> None:
    visualizeDataset(logger)