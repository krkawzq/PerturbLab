__version__ = "0.2.4"
import logging
import sys

logger = logging.getLogger("scGPT")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from . import model, scbank, tasks, tokenizer, utils
from .data_collator import DataCollator
from .data_sampler import SubsetsBatchSampler
from .trainer import (define_wandb_metrcis, eval_testdata, evaluate,
                      prepare_data, prepare_dataloader, test, train)
