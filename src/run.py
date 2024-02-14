import time
import importlib
import logging
import argparse
from argparse import RawTextHelpFormatter

logger = logging.getLogger("FSD")


def run_experiment(args, logger):
    if args.step == 1:
        main_module = "Visualization.main"
    elif args.step == 2:
        main_module = "LDA.main"
    elif args.step == 3:
        main_module = "Dimensionality.main"
    elif args.step == 4:
        main_module = "LDA.main"
    elif args.step == 5:
        main_module = "GaussianClassifiers.main"
    elif args.step == 6:
        main_module = "LogisticRegression.main"
    elif args.step == 7:
        main_module = "SVM.main"
    elif args.step == 8:
        main_module = "GMM.main"

    else:
        logger.error("Invalid step")
        raise NotImplementedError

    main = getattr(importlib.import_module(main_module), "main")
    main(args, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Fingerprint Spoofing Detection",
        description="Based on the choosen step the program will perform different actions.",
        epilog="Choose a Step and run the program.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        help="Step to run:\n"
        + "\t1: Visualization\n"
        + "\t2: LDA\n"
        + "\t3: Dimensionality Reduction\n"
        + "\t4: LDA\n"
        + "\t5: MVG\n"
        + "\t6: Logistic Regression\n"
        + "\t7: SVM\n"
        + "\t8: GMM\n",
        required=True,
    )
    parser.add_argument(
        "-k",
        type=int,
        help="Number of folds for k-fold cross validation",
        default=5,
        required=False,
    )
    parser.add_argument(
        "-m",
        type=int,
        help="used for LDA ",
        default=1,
        required=False,
    )
    parser.add_argument(
        "-pca",
        type=int,
        help="number of eigenvectors to use for PCA",
        default=6,
        required=False,
    )
    parser.add_argument(
        "-znorm",
        type=bool,
        help="Enable Z-Normalization",
        default=False,
        required=False,
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    start = time.time()

    run_experiment(args, logger)

    end = time.time()
    logger.info(f"Elapsed time: {round(end - start, 2)}")
