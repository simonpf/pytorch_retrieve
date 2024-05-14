"""
pytorch_retrieve.cli
====================

Defines the pytorch_retrieve command line interface.
"""
import logging

import click
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
    force=True
)

from pytorch_retrieve import eda
from pytorch_retrieve import lr_search
from pytorch_retrieve import training
from pytorch_retrieve import inference


@click.group()
def pytorch_retrieve():
    """
    Welcome to the pytorch_retrieve command-line interface.
    """
    pass


pytorch_retrieve.command(name="eda")(eda.cli)
pytorch_retrieve.command(name="lr_search")(lr_search.cli)
pytorch_retrieve.command(name="train")(training.cli)
pytorch_retrieve.command(name="inference")(inference.cli)
