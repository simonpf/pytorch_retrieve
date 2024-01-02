"""
pytorch_retrieve.modules.utils
==============================

Utilities for the pytorch_retrieve.modules module.
"""


class ParamCount:
    """
    Mixin class for pytorch modules that add a 'n_params' attribute
    to the class.
    """

    @property
    def n_params(self):
        """
        The number of trainable parameters in the network.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
