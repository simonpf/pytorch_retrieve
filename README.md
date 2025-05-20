# PyTorch Retrieve 🛰️

Neural-network-based remote-sensing retrievals (and more) for the busy remote-sensing scientist. 👩‍🔬

## Why PyTorch Retrieve?

The aim of PyTorch Retrieve is to provide remote-sensing scientists with a fast,
fail-safe and flexible way of training retrievals of geophysical quantities from
satellite observations. PyTorch Retrieve's principal features are

1. flexible implementations of state-of-the-art neural network architecture that
   can be trained on a wide range of input data including multi-spectral,
   multi-sensor and multi time step data,
2. multi-output retrievals handling scalar, vector, continuous and catergorical outputs,
3. modular model configuration using configuration files in '.toml' or '.yaml' format,
4. probabilistic regression using quantiles or binned distributions,
5. built-in handling of input normalization, value imputation, and output masking.

## PyTorch Retrieve vs. other packages for geo-spatial DL

Why another deep-learning package for satellite data?

The other deep-learning pacakges for geospatial data that I am aware of  ([TorchGeo](https://github.com/microsoft/torchgeo) and [TorchSat](https://github.com/sshuair/torchsat)) were designed with classification tasks in mind and most of their functionality focuses on loading geospatial data or providing interfaces to existing geospatial ML datasets. PyTorch retrieve focuses on dense quantification tasks, i.e. predicting scalar or vector quantities for every or almost every pixel in the input data.

PyTorch Retrieve takes a different approach in the functionality it offers.
Instead of focusing on simplifying data loading, it aims to make it easier to
implement a well-performing neural network. The goal is to separate the
scientific aspects (preparing training data and evaluating retrieval
performance) from the engineering side of things, like training the model architecture
and training recipe. By keeping these parts separate, changing the neural network
architecture becomes as simple as modifying a configuration file.
