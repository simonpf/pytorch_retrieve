"""
pytorch_retrieve.cli
====================

Defines the pytorch_retrieve command line interface.
"""
import click

from pytorch_retrieve import eda
from pytorch_retrieve import lr_search
from pytorch_retrieve import training


@click.group()
def pytorch_retrieve():
    """
    Welcome to the pytorch_retrieve command-line interface.
    """
    pass


pytorch_retrieve.command(name="eda")(eda.cli)
pytorch_retrieve.command(name="lr_search")(lr_search.cli)
pytorch_retrieve.command(name="train")(training.cli)
