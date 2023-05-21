
from Utils.utils import load


def main(args,logger):
    logger.info("Starting main function on LGA")
    D,C = load()
    logger.info(f"Loaded {D.shape[0]} samples with {D.shape[1]} features")
    logger.info(f"Loaded {C.shape[0]} classes")
    logger.info("example of sample:")
    logger.info(D[0])
    logger.info("Loaded data")