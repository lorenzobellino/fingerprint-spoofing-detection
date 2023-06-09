import time
import importlib
import logging
import argparse
from argparse import RawTextHelpFormatter

logger = logging.getLogger("FSD")

def run_experiment(args, logger):
    if args.step == 1:
        main_module = "LDA.main"
    elif args.step == 2:
        main_module = "PCA.main"
    elif args.step == 3:
        main_module = "MGD.main"
    elif args.step == 4:
        main_module = "MLE.main"
    else:
        logger.error("Invalid step")
        raise NotImplementedError
   
    main = getattr(importlib.import_module(main_module), "main")
    main(args, logger)

if __name__ == '__main__':
    
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
        + "\t1: LDA - Linear Discriminant Analysis\n"
        + "\t2: PCA - Principal Component Analysis\n"
        + "\t3: MGD - Multivariate Gaussian Density\n"
        + "\t4: MLE - Maximum Likelihood Estimate\n",
        required=True,
    )
    parser.add_argument(
        "-m",
        type=int,
        help="Number of eigenvectors to use for PCA and LDA",
        default=4,
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