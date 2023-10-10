from Utils.utils import *

def main(args, logger) -> None:
    logger.info(
        "\n###########################################################\n"+
        "#                                                         #\n"+
        "#                    VISUALIZATION                        #\n"+
        "#                                                         #\n"+
        "###########################################################")
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
    logger.info("Step Completed, you can find the plots in the plots folder")