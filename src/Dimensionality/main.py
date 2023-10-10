from Utils.utils import *
from PCA.main import PCA

def main(args,logger) -> None:
    logger.info(
        "\n###########################################################\n"+
        "#                                                         #\n"+
        "#              DIMENSIONALITY REDUCTION                   #\n"+
        "#                                                         #\n"+
        "###########################################################")
    D,C = load(logger)
    DTR , LTR = randomize(D, C)
    logger.info("Plotting Explained Variance")
    logger.info("Plotting Scree Plot")
    plot_explained_variance(logger,DTR,PCA)
    # input()
    logger.info("TESTING FOR DIFFERENT m VALUES")
    logger.info("Splitting the dataset for testing and validation")
    # DTR, DTE, LTR, LTE = split_db_2to1(D, C)
    DTR, DTE, LTR, LTE = split_db_n(D, C, n=1/3)
    logger.debug(f"DTR shape: {DTR.shape} DTE shape: {DTE.shape} LTR shape: {LTR.shape} LTE shape: {LTE.shape}")
    logger.info("\n-------------------PCA---------------------\n")
    for m in range(10,1,-1):
        logger.info("--------------------------------")
        logger.info(f"Testing for m = {m}")
        P,_ = PCA(logger, DTR, m)
        # projection of the training set
        DTR_P = np.dot(P.T, DTR)
        # projection of the evaluation set
        DTE_P = np.dot(P.T, DTE)
        # mean of the training set
        mean_original = np.mean(DTR_P[:, LTR == 1], axis=1)
        mean_spoofed = np.mean(DTR_P[:, LTR == 0], axis=1)
        # prediction
        prediction = [1 if np.linalg.norm(DTE_P[:, i]-mean_original)**2 < np.linalg.norm(DTE_P[:, i]-mean_spoofed)**2 else 0 for i in range(DTE_P.shape[1])]
        # accuracy
        accuracy = np.sum(prediction == LTE) / LTE.shape[0]
        logger.info(f"Accuracy: {accuracy*100:.2f}%")





        
