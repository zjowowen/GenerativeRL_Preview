import wandb
import logging
from rich.logging import RichHandler
import os

# Silence wandb by using the following line
# os.environ["WANDB_SILENT"] = "True"
# wandb_logger = logging.getLogger("wandb")
# wandb_logger.setLevel(logging.ERROR)

log = logging.getLogger("rich")
