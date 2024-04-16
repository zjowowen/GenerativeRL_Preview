import logging
import os

import wandb
from rich.logging import RichHandler

# Silence wandb by using the following line
# os.environ["WANDB_SILENT"] = "True"
# wandb_logger = logging.getLogger("wandb")
# wandb_logger.setLevel(logging.ERROR)

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")
