from torch.utils.data import DataLoader

from pytorch_retrieve.data.synthetic import Synthetic1d, Synthetic1dMultiOutput


def data_loader_1d(n_samples: int, batch_size: int) -> DataLoader:
    """
    A DataLoader providing batch of Synthetic1D data.
    """
    data = Synthetic1d(n_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)
