"""
pytorch_retrieve.data.synthetic
===============================

Defines data set class providing synthetic data for testing purposes.
"""
from math import ceil
from typing import Tuple

import numpy as np
from scipy.fft import idctn

import torch


def random_spectral_field(size, bands, max_wave_nbr=1 / 4):
    """
    Create a random spectral field and split the data into spectral bands.


    Args:
        size: Tuple ``(h, w)`` specifying the height and widt of the field.
        bands: The number of bands of the output.

    Return:
        A tuple ``(y, y_bands)`` containing random 2D field and ``y_bands``
        where the 2D field y is split up into frequency bands.
    """
    wny = 0.5 * np.arange(size[0]) / size[0]
    wnx = 0.5 * np.arange(size[1]) / size[1]
    wn = np.sqrt(wny[..., None] ** 2 + wnx**2)

    bins = np.linspace(0, wn.max() * max_wave_nbr, bands + 1)

    field_tot = np.zeros(size, np.float32)
    fields = []

    for ind in range(bins.size - 1):
        min_wn = bins[ind]
        max_wn = bins[ind + 1]

        coeffs = np.random.rand(*size).astype(np.float32) - 0.5
        coeffs[0, 0] = 0.0
        coeffs[wn < min_wn] = 0.0
        coeffs[wn >= max_wn] = 0.0

        field_i = idctn(coeffs, norm="ortho")

        if field_tot is None:
            field_tot = field_i
        else:
            field_tot += field_i
        fields.append(field_i)

    field = np.stack(fields)

    return field_tot, field


class Synthetic1d:
    """
    A data loader for synthetic, tabular data with heteroscedastic noise.
    """

    def __init__(self, n_samples):
        """
        Args:
            n_samples: The number of samples in the training data.
        """
        self.n_samples = n_samples

        if n_samples < 128:
            raise ValueError("Sample should be at least 128.")

        y_i, y_bands_i = random_spectral_field((1, 128), bands=4, max_wave_nbr=1 / 16)
        x_i = np.arange(0, 128) / 128

        x = np.random.rand(n_samples)

        y = np.interp(x, x_i, y_i[0])
        scl = np.interp(x, x_i, y_bands_i[1, 0])

        err = scl * np.random.normal(size=n_samples)
        y = y + err

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        """The number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, index):
        """
        Return a sample from the dataset.

        Args:
            index: The index of the sample in the dataset.

        Return:
            A tuple ``(x, y)`` containing the input ``x`` and corresponding
            prediction target ``y``.
        """
        return (torch.tensor(self.x[index, None]), torch.tensor(self.y[index, None]))


class Synthetic1dMultiInput(Synthetic1d):
    """
    An extension of the Synthetic1d dataset that provides two input features
    named "x_1" and "x_2".
    """

    def __init__(self, n_samples=2045):
        """
        Args:
            n_samples: The number of samples in the training data.
        """
        super().__init__(n_samples=n_samples)

    def __getitem__(self, index):
        """
        Return a sample from the dataset.

        Args:
            index: The index of the sample in the dataset.

        Return:
            A tuple ``(x, ys)`` containing the input ``x`` and corresponding
            prediction target ``ys``, where ``ys`` is a dictionary mapping the
            output names ``y`` and ``-y`` to corresponding outputs.
        """
        x = torch.tensor(self.x[index, None])
        y = torch.tensor(self.y[index, None])
        return ({"x_1": x, "x_2": x}, y)


class Synthetic1dMultiOutput(Synthetic1d):
    """
    An extension of the Synthetic1d dataset that provides two output targets
    'y' and '-y'.
    """

    def __init__(self, n_samples=2045):
        """
        Args:
            n_samples: The number of samples in the training data.
        """
        super().__init__(n_samples=n_samples)

    def __getitem__(self, index):
        """
        Return a sample from the dataset.

        Args:
            index: The index of the sample in the dataset.

        Return:
            A tuple ``(x, ys)`` containing the input ``x`` and corresponding
            prediction target ``ys``, where ``ys`` is a dictionary mapping the
            output names ``y`` and ``-y`` to corresponding outputs.
        """
        x = torch.tensor(self.x[index, None])
        y = torch.tensor(self.y[index, None])
        return (x, {"y": y, "-y": -y})


class Synthetic3d:
    """
    A data loader for synthetic, multi-spectral image data with heteroscedastic
    noise.
    """

    def __init__(self, n_samples=2045):
        """
        Args:
            n_samples: The number of samples in the training data.
        """
        self.n_samples = n_samples

        samples_x = []
        samples_y = []
        for _ in range(n_samples):
            y_i, x_i = random_spectral_field((128, 128), bands=4, max_wave_nbr=1 / 16)
            samples_x.append(x_i)
            samples_y.append(y_i)

        x = np.stack(samples_x, axis=0)
        y = np.stack(samples_y, axis=0)

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        """The number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, index):
        """
        Return a sample from the dataset.

        Args:
            index: The index of the sample in the dataset.

        Return:
            A tuple ``(x, y)`` containing the input ``x`` and corresponding
            prediction target ``y``.
        """
        return (torch.tensor(self.x[index]), torch.tensor(self.y[index]))
